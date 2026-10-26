using UnityEngine;

public class VehicleController : MonoBehaviour
{
    private Rigidbody rb;

    private Vector3 lastPos;
    private Quaternion lastRot;
    private Vector3 curPos;
    private Quaternion curRot;
    private float lastTime;
    private float curTime;

    private float curLong, curVert, curLat;
    // set at runtime, after the Inspector value is known
    private float stepLen;
    private float turnThresholdDeg;

    private const float FadeTime = 0.05f;          // how long to ease out spin

    private Vector3 residualAngularVel;           // ★ keeps turn's leftover spin
    private float residualTimer;                // ★ fade-out countdown

    [Header("Dataset / Occlusion")]
    [Tooltip("Debe existir en Project Settings > Tags and Layers, o este vehículo no podrá ocluir a otros.")]
    [SerializeField] private string occlusionLayerName = "DatasetOcclusion";

    private void Start()
    {
        rb = GetComponent<Rigidbody>() ?? gameObject.AddComponent<Rigidbody>();

        rb.isKinematic = false;
        rb.useGravity = false;
        rb.linearDamping = 1f;
        rb.interpolation = RigidbodyInterpolation.Interpolate;
        rb.collisionDetectionMode = CollisionDetectionMode.Continuous;

        curPos = lastPos = transform.position;
        curRot = lastRot = transform.rotation;
        lastTime = curTime = Time.time;

        SetupOcclusionCollider();
    }

    // ============================================================
    // OCCLUSION COLLIDER (para YoloDatasetGenerator)
    // ============================================================

    private void SetupOcclusionCollider()
    {
        int layer = LayerMask.NameToLayer(occlusionLayerName);

        if (layer == -1)
        {
            Debug.LogWarning(
                $"VehicleController: la layer '{occlusionLayerName}' no existe. " +
                "Créala en Edit > Project Settings > Tags and Layers. " +
                "Sin ella, este vehículo no podrá tapar a otros en el dataset."
            );
            return;
        }

        Renderer[] renderers = GetComponentsInChildren<Renderer>();

        if (renderers == null || renderers.Length == 0)
        {
            return;
        }

        Bounds worldBounds = renderers[0].bounds;

        for (int i = 1; i < renderers.Length; i++)
        {
            worldBounds.Encapsulate(renderers[i].bounds);
        }

        // GameObject hijo dedicado: así el layer del vehículo raíz
        // (usado para renderizado/culling) queda intacto.
        GameObject occlusionVolume = new GameObject("OcclusionVolume");
        occlusionVolume.layer = layer;
        occlusionVolume.transform.SetParent(transform, worldPositionStays: false);
        occlusionVolume.transform.position = worldBounds.center;
        occlusionVolume.transform.rotation = transform.rotation;

        BoxCollider occlusionCollider = occlusionVolume.AddComponent<BoxCollider>();
        occlusionCollider.isTrigger = true;

        Vector3 lossyScale = occlusionVolume.transform.lossyScale;

        occlusionCollider.size = new Vector3(
            lossyScale.x != 0 ? worldBounds.size.x / lossyScale.x : worldBounds.size.x,
            lossyScale.y != 0 ? worldBounds.size.y / lossyScale.y : worldBounds.size.y,
            lossyScale.z != 0 ? worldBounds.size.z / lossyScale.z : worldBounds.size.z
        );
    }

    public void UpdateTarget(Vector3 pos, Quaternion rot,
                             float longSpd, float vertSpd, float latSpd)
    {
        lastPos = curPos; lastRot = curRot; lastTime = curTime;
        curPos = pos; curRot = rot; curTime = Time.time;

        curLong = longSpd; curVert = vertSpd; curLat = latSpd;
    }
    void Awake()
    {
        // Look for the first SimulationController in the scene
        SimulationController sim = FindObjectOfType<SimulationController>();

        if (sim == null)
        {
            Debug.LogError("SimulationController not found!");
            return;
        }

        stepLen = sim.unityStepLength;                 // ← value set in Inspector
        turnThresholdDeg = Mathf.Clamp(stepLen * 40f, 0.25f, 10f);
    }
    private void FixedUpdate()
    {
        float dt = curTime - lastTime;
        if (dt <= 0f)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            rb.MoveRotation(curRot);
            return;
        }

        float headingDelta = Quaternion.Angle(lastRot, curRot);   // degrees

        if (headingDelta < turnThresholdDeg)                      // ─ straight
        {
            /* linear vel from local-axis speeds (ultra smooth) */
            Vector3 vLong = curRot * (Vector3.right * curLong);
            Vector3 vLat = curRot * (Vector3.forward * curLat);
            Vector3 vUp = Vector3.up * curVert;
            rb.linearVelocity = vLong + vLat + vUp;

            /* ① damp residual spin, don't kill instantly */
            if (residualTimer > 0f)
            {
                residualTimer -= Time.fixedDeltaTime;
                float k = Mathf.Clamp01(residualTimer / FadeTime);
                rb.angularVelocity = residualAngularVel * k;
                if (k <= 0f) rb.MoveRotation(curRot);            // fully aligned
            }
            else
            {
                rb.angularVelocity = Vector3.zero;
                rb.MoveRotation(curRot);
            }
        }
        else                                                      // ─ turning
        {
            rb.linearVelocity = (curPos - lastPos) / dt;
            residualAngularVel = CalcAngularVel(lastRot, curRot, dt); // ★ store
            rb.angularVelocity = residualAngularVel;
            residualTimer = FadeTime;                          // ★ reset
        }

        /* original ultra-smooth positional blend */
        transform.localPosition =
            Vector3.Lerp(transform.localPosition, curPos, 0.02f);
    }

    private static Vector3 CalcAngularVel(Quaternion from, Quaternion to, float dt)
    {
        Quaternion dq = to * Quaternion.Inverse(from);
        dq.ToAngleAxis(out float angDeg, out Vector3 axis);
        if (angDeg > 180f) angDeg -= 360f;
        return axis.normalized * Mathf.Deg2Rad * angDeg / Mathf.Max(dt, 0.0001f);
    }
}