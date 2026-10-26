using UnityEngine;

public class BoundingBoxVisualizer : MonoBehaviour
{
    [Header("Camera")]
    public Camera targetCamera;

    [Header("Dataset Resolution")]
    public int imageWidth = 1920;
    public int imageHeight = 1080;

    [Header("Visualization")]
    public bool showBoundingBoxes = true;
    public bool showLabels = true;

    public float lineThickness = 3f;

    private Texture2D lineTexture;

    private void Start()
    {
        if (targetCamera == null)
            targetCamera = Camera.main;

        lineTexture = new Texture2D(1, 1);
        lineTexture.SetPixel(0, 0, Color.red);
        lineTexture.Apply();
    }

    private void OnGUI()
    {
        if (!showBoundingBoxes)
            return;

        if (targetCamera == null)
            return;

        VehicleGroundTruth[] vehicles =
            FindObjectsByType<VehicleGroundTruth>(
                FindObjectsSortMode.None
            );

        foreach (VehicleGroundTruth vehicle in vehicles)
        {
            if (vehicle == null)
                continue;

            if (!YoloBoundingBoxUtility.TryGetBoundingBox(
                    vehicle.gameObject,
                    targetCamera,
                    out YoloBoundingBoxUtility.BoundingBox box))
            {
                continue;
            }

            DrawBoundingBox(box, vehicle);
        }
    }

    private void DrawBoundingBox(
        YoloBoundingBoxUtility.BoundingBox box,
        VehicleGroundTruth vehicle)
    {
        // Convert normalized coordinates to screen coordinates.
        float x = box.xMin * Screen.width;

        float y = (1f - box.yMax) * Screen.height;

        float width = box.Width * Screen.width;
        float height = box.Height * Screen.height;

        Rect rect = new Rect(
            x,
            y,
            width,
            height
        );

        DrawRect(rect);

        if (showLabels)
        {
            string label =
                vehicle.vehicleType.ToString();

            if (vehicle.vehicleId != 0)
            {
                label += " #" + vehicle.vehicleId;
            }

            GUI.Label(
                new Rect(
                    rect.x,
                    rect.y - 22,
                    250,
                    22
                ),
                label
            );
        }
    }

    private void DrawRect(Rect rect)
    {
        // Top
        GUI.DrawTexture(
            new Rect(
                rect.x,
                rect.y,
                rect.width,
                lineThickness
            ),
            lineTexture
        );

        // Bottom
        GUI.DrawTexture(
            new Rect(
                rect.x,
                rect.y + rect.height - lineThickness,
                rect.width,
                lineThickness
            ),
            lineTexture
        );

        // Left
        GUI.DrawTexture(
            new Rect(
                rect.x,
                rect.y,
                lineThickness,
                rect.height
            ),
            lineTexture
        );

        // Right
        GUI.DrawTexture(
            new Rect(
                rect.x + rect.width - lineThickness,
                rect.y,
                lineThickness,
                rect.height
            ),
            lineTexture
        );
    }
}