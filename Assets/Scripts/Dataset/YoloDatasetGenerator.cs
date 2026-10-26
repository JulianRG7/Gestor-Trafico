using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

public class YoloDatasetGenerator : MonoBehaviour
{
    [Header("Capture Settings")]
    [SerializeField] private Camera captureCamera;

    [SerializeField] private int imageWidth = 1920;
    [SerializeField] private int imageHeight = 1080;

    [SerializeField] private int numberOfImages = 100;
    [SerializeField] private float captureInterval = 1.0f;

    [SerializeField] private string datasetFolderName = "YOLODataset";

    [SerializeField] private bool generateOnStart = false;

    [Header("Occlusion / Visibility")]
    [Tooltip("Enable visibility checking before creating YOLO labels.")]
    [SerializeField] private bool enableOcclusionCheck = true;

    [Range(0.0f, 1.0f)]
    [Tooltip("Minimum percentage of sampled points that must be visible.")]
    [SerializeField] private float minimumVisibility = 0.30f;

    [Range(5, 27)]
    [Tooltip("Number of points sampled from the vehicle Bounds.")]
    [SerializeField] private int samplePoints = 9;

    [Tooltip("Maximum distance used by the visibility raycast.")]
    [SerializeField] private float maxRayDistance = 1000f;

    [Header("Debug")]
    [SerializeField] private bool logVehicleInformation = true;

    [SerializeField] private bool drawVisibilityRays = false;

    // ============================================================
    // INTERNAL
    // ============================================================

    private RenderTexture renderTexture;
    private Texture2D screenTexture;

    private string datasetRoot;
    private string imagesFolder;
    private string labelsFolder;

    private bool isGenerating = false;
    private int currentImageIndex = 0;

    // ============================================================
    // UNITY
    // ============================================================

    private void Start()
    {
        if (generateOnStart)
        {
            StartDatasetGeneration();
        }
    }

    // ============================================================
    // PUBLIC
    // ============================================================

    public void StartDatasetGeneration()
    {
        if (isGenerating)
        {
            Debug.LogWarning(
                "Dataset generation is already running."
            );

            return;
        }

        if (captureCamera == null)
        {
            Debug.LogError(
                "YoloDatasetGenerator: Capture Camera is not assigned."
            );

            return;
        }

        if (imageWidth <= 0 || imageHeight <= 0)
        {
            Debug.LogError(
                "YoloDatasetGenerator: Invalid image dimensions."
            );

            return;
        }

        CreateDatasetFolders();

        PrepareCaptureResources();

        currentImageIndex = 0;

        isGenerating = true;

        Debug.Log(
            $"Starting dataset generation: " +
            $"{imageWidth}x{imageHeight}"
        );

        Debug.Log(
            $"Images: {numberOfImages} | " +
            $"Interval: {captureInterval}s | " +
            $"Minimum Visibility: {minimumVisibility:P0} | " +
            $"Sample Points: {samplePoints}"
        );

        StartCoroutine(GenerateDatasetCoroutine());
    }

    public void StopDatasetGeneration()
    {
        if (!isGenerating)
        {
            return;
        }

        StopAllCoroutines();

        isGenerating = false;

        Debug.Log("Dataset generation manually stopped.");
    }

    // ============================================================
    // DATASET
    // ============================================================

    private IEnumerator GenerateDatasetCoroutine()
    {
        while (currentImageIndex < numberOfImages)
        {
            yield return new WaitForEndOfFrame();

            GenerateSingleImage(currentImageIndex);

            currentImageIndex++;

            if (currentImageIndex < numberOfImages)
            {
                yield return new WaitForSeconds(captureInterval);
            }
        }

        isGenerating = false;

        Debug.Log(
            $"Dataset generation finished. " +
            $"Images: {currentImageIndex}"
        );
    }

    private void GenerateSingleImage(int frameIndex)
    {
        // --------------------------------------------------------
        // 1. Render camera
        // --------------------------------------------------------

        RenderTexture previousTarget =
            captureCamera.targetTexture;

        RenderTexture previousActive =
            RenderTexture.active;

        captureCamera.targetTexture = renderTexture;

        captureCamera.Render();

        RenderTexture.active = renderTexture;

        // --------------------------------------------------------
        // 2. Read image
        // --------------------------------------------------------

        screenTexture.ReadPixels(
            new Rect(
                0,
                0,
                imageWidth,
                imageHeight
            ),
            0,
            0
        );

        screenTexture.Apply();

        byte[] imageBytes =
            screenTexture.EncodeToPNG();

        // --------------------------------------------------------
        // 3. Restore camera
        // --------------------------------------------------------

        captureCamera.targetTexture = previousTarget;

        RenderTexture.active = previousActive;

        // --------------------------------------------------------
        // 4. Get vehicles
        // --------------------------------------------------------

        List<VehicleGroundTruth> vehicles =
            VehicleGroundTruth.AllVehicles;

        StringBuilder labels =
            new StringBuilder();

        int visibleVehicles = 0;
        int occludedVehicles = 0;
        int invalidVehicles = 0;

        // --------------------------------------------------------
        // 5. Process vehicles
        // --------------------------------------------------------

        for (int i = 0; i < vehicles.Count; i++)
        {
            VehicleGroundTruth vehicle =
                vehicles[i];

            if (vehicle == null)
            {
                continue;
            }

            if (!vehicle.gameObject.activeInHierarchy)
            {
                continue;
            }

            // ----------------------------------------------------
            // Bounding Box
            // ----------------------------------------------------

            YoloBoundingBoxUtility.BoundingBox box;

            bool hasBoundingBox =
                YoloBoundingBoxUtility.TryGetBoundingBox(
                    vehicle.gameObject,
                    captureCamera,
                    out box
                );

            if (!hasBoundingBox)
            {
                invalidVehicles++;
                continue;
            }

            // ----------------------------------------------------
            // Occlusion
            // ----------------------------------------------------

            float visibility = 1.0f;

            if (enableOcclusionCheck)
            {
                visibility =
                    CalculateVisibility(vehicle);
            }

            if (enableOcclusionCheck &&
                visibility < minimumVisibility)
            {
                occludedVehicles++;

                if (logVehicleInformation)
                {
                    Debug.Log(
                        $"Vehicle {vehicle.vehicleId} " +
                        $"({vehicle.vehicleType}) discarded. " +
                        $"Visibility: {visibility:P1}"
                    );
                }

                continue;
            }

            visibleVehicles++;

            // ----------------------------------------------------
            // YOLO label
            // ----------------------------------------------------

            string yoloLine =
                YoloBoundingBoxUtility.ToYoloString(
                    vehicle.vehicleType,
                    box
                );

            labels.AppendLine(yoloLine);

            if (logVehicleInformation)
            {
                Debug.Log(
                    $"Vehicle {vehicle.vehicleId} " +
                    $"({vehicle.vehicleType}) included. " +
                    $"Visibility: {visibility:P1}"
                );
            }
        }

        // --------------------------------------------------------
        // 6. Save image
        // --------------------------------------------------------

        string fileName =
            $"image_{frameIndex:D06}";

        string imagePath =
            Path.Combine(
                imagesFolder,
                fileName + ".png"
            );

        string labelPath =
            Path.Combine(
                labelsFolder,
                fileName + ".txt"
            );

        File.WriteAllBytes(
            imagePath,
            imageBytes
        );

        File.WriteAllText(
            labelPath,
            labels.ToString()
        );

        Debug.Log(
            $"Generated {fileName} | " +
            $"Vehicles included: {visibleVehicles} | " +
            $"Occluded: {occludedVehicles} | " +
            $"Invalid: {invalidVehicles}"
        );
    }

    // ============================================================
    // OCCLUSION
    // ============================================================

    private float CalculateVisibility(
        VehicleGroundTruth vehicle
    )
    {
        Renderer[] renderers =
            vehicle.GetComponentsInChildren<Renderer>();

        if (renderers == null ||
            renderers.Length == 0)
        {
            return 0f;
        }

        Bounds bounds =
            renderers[0].bounds;

        for (int i = 1; i < renderers.Length; i++)
        {
            bounds.Encapsulate(
                renderers[i].bounds
            );
        }

        Vector3[] points =
            GenerateSamplePoints(bounds);

        int visiblePoints = 0;

        for (int i = 0; i < points.Length; i++)
        {
            if (IsPointVisible(
                vehicle,
                points[i]
            ))
            {
                visiblePoints++;
            }
        }

        if (points.Length == 0)
        {
            return 0f;
        }

        return
            (float)visiblePoints /
            points.Length;
    }

    private Vector3[] GenerateSamplePoints(
        Bounds bounds
    )
    {
        List<Vector3> points =
            new List<Vector3>();

        Vector3 min = bounds.min;
        Vector3 max = bounds.max;
        Vector3 center = bounds.center;

        // --------------------------------------------------------
        // Always include center
        // --------------------------------------------------------

        points.Add(center);

        // --------------------------------------------------------
        // Eight corners
        // --------------------------------------------------------

        points.Add(
            new Vector3(min.x, min.y, min.z)
        );

        points.Add(
            new Vector3(min.x, min.y, max.z)
        );

        points.Add(
            new Vector3(min.x, max.y, min.z)
        );

        points.Add(
            new Vector3(min.x, max.y, max.z)
        );

        points.Add(
            new Vector3(max.x, min.y, min.z)
        );

        points.Add(
            new Vector3(max.x, min.y, max.z)
        );

        points.Add(
            new Vector3(max.x, max.y, min.z)
        );

        points.Add(
            new Vector3(max.x, max.y, max.z)
        );

        // --------------------------------------------------------
        // Additional points for more accurate sampling
        // --------------------------------------------------------

        if (samplePoints >= 13)
        {
            points.Add(
                new Vector3(
                    center.x,
                    min.y,
                    center.z
                )
            );

            points.Add(
                new Vector3(
                    center.x,
                    max.y,
                    center.z
                )
            );

            points.Add(
                new Vector3(
                    min.x,
                    center.y,
                    center.z
                )
            );

            points.Add(
                new Vector3(
                    max.x,
                    center.y,
                    center.z
                )
            );
        }

        if (samplePoints >= 17)
        {
            points.Add(
                new Vector3(
                    center.x,
                    center.y,
                    min.z
                )
            );

            points.Add(
                new Vector3(
                    center.x,
                    center.y,
                    max.z
                )
            );

            points.Add(
                new Vector3(
                    min.x,
                    center.y,
                    min.z
                )
            );

            points.Add(
                new Vector3(
                    max.x,
                    center.y,
                    max.z
                )
            );
        }

        if (samplePoints >= 27)
        {
            float x1 = Mathf.Lerp(
                min.x,
                max.x,
                0.5f
            );

            float y1 = Mathf.Lerp(
                min.y,
                max.y,
                0.5f
            );

            float z1 = Mathf.Lerp(
                min.z,
                max.z,
                0.5f
            );

            points.Add(
                new Vector3(x1, min.y, min.z)
            );

            points.Add(
                new Vector3(x1, min.y, max.z)
            );

            points.Add(
                new Vector3(x1, max.y, min.z)
            );

            points.Add(
                new Vector3(x1, max.y, max.z)
            );

            points.Add(
                new Vector3(min.x, y1, min.z)
            );

            points.Add(
                new Vector3(min.x, y1, max.z)
            );

            points.Add(
                new Vector3(max.x, y1, min.z)
            );

            points.Add(
                new Vector3(max.x, y1, max.z)
            );

            points.Add(
                new Vector3(min.x, min.y, z1)
            );

            points.Add(
                new Vector3(min.x, max.y, z1)
            );

            points.Add(
                new Vector3(max.x, min.y, z1)
            );

            points.Add(
                new Vector3(max.x, max.y, z1)
            );
        }

        return points.ToArray();
    }

    private bool IsPointVisible(
        VehicleGroundTruth vehicle,
        Vector3 point
    )
    {
        Vector3 origin =
            captureCamera.transform.position;

        Vector3 direction =
            point - origin;

        float distance =
            direction.magnitude;

        if (distance <= 0.001f)
        {
            return true;
        }

        direction.Normalize();
        RaycastHit[] hits =
            Physics.RaycastAll(
                origin,
                direction,
                Mathf.Min(
                    distance,
                    maxRayDistance
                ),
                Physics.DefaultRaycastLayers,
                QueryTriggerInteraction.Collide   // ← antes: QueryTriggerInteraction.Ignore
            );
        if (hits == null ||
            hits.Length == 0)
        {
            return true;
        }

        Array.Sort(
            hits,
            (a, b) =>
                a.distance.CompareTo(b.distance)
        );

        for (int i = 0; i < hits.Length; i++)
        {
            Collider hitCollider =
                hits[i].collider;

            if (hitCollider == null)
            {
                continue;
            }

            // ----------------------------------------------------
            // Is the first relevant object part of this vehicle?
            // ----------------------------------------------------

            if (
                hitCollider.transform == vehicle.transform ||
                hitCollider.transform.IsChildOf(
                    vehicle.transform
                )
            )
            {
                if (drawVisibilityRays)
                {
                    Debug.DrawRay(
                        origin,
                        direction * distance,
                        Color.green,
                        0.1f
                    );
                }

                return true;
            }

            // ----------------------------------------------------
            // Something else is between the camera and point
            // ----------------------------------------------------

            if (drawVisibilityRays)
            {
                Debug.DrawRay(
                    origin,
                    direction * distance,
                    Color.red,
                    0.1f
                );
            }

            return false;
        }

        return true;
    }

    // ============================================================
    // FOLDERS
    // ============================================================

    private void CreateDatasetFolders()
    {
        string projectRoot =
            Directory.GetParent(
                Application.dataPath
            ).FullName;

        datasetRoot =
            Path.Combine(
                projectRoot,
                datasetFolderName
            );

        imagesFolder =
            Path.Combine(
                datasetRoot,
                "images"
            );

        labelsFolder =
            Path.Combine(
                datasetRoot,
                "labels"
            );

        Directory.CreateDirectory(
            imagesFolder
        );

        Directory.CreateDirectory(
            labelsFolder
        );

        Debug.Log(
            $"Dataset folder: {datasetRoot}"
        );
    }

    // ============================================================
    // CAPTURE RESOURCES
    // ============================================================

    private void PrepareCaptureResources()
    {
        if (renderTexture != null)
        {
            renderTexture.Release();

            Destroy(
                renderTexture
            );
        }

        if (screenTexture != null)
        {
            Destroy(
                screenTexture
            );
        }

        renderTexture =
            new RenderTexture(
                imageWidth,
                imageHeight,
                24,
                RenderTextureFormat.ARGB32
            );

        renderTexture.Create();

        screenTexture =
            new Texture2D(
                imageWidth,
                imageHeight,
                TextureFormat.RGB24,
                false
            );

        captureCamera.aspect =
            (float)imageWidth /
            imageHeight;
    }

    // ============================================================
    // CLEANUP
    // ============================================================

    private void OnDestroy()
    {
        if (renderTexture != null)
        {
            renderTexture.Release();

            Destroy(
                renderTexture
            );
        }

        if (screenTexture != null)
        {
            Destroy(
                screenTexture
            );
        }
    }
}