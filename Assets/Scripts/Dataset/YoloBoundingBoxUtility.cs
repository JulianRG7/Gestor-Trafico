using System.Globalization;
using UnityEngine;

public static class YoloBoundingBoxUtility
{
    public struct BoundingBox
    {
        public float xMin;
        public float yMin;
        public float xMax;
        public float yMax;

        public float Width => xMax - xMin;
        public float Height => yMax - yMin;
        public float CenterX => (xMin + xMax) / 2f;
        public float CenterY => (yMin + yMax) / 2f;
    }

    // Buffer estático reutilizable de 8 esquinas para evitar Garbage Collection
    private static readonly Vector3[] CornersBuffer = new Vector3[8];

    public static bool TryGetBoundingBox(
        GameObject vehicle,
        Camera camera,
        out BoundingBox box,
        Renderer[] preCachedRenderers = null)
    {
        box = new BoundingBox();

        if (vehicle == null || camera == null)
        {
            return false;
        }

        // Utilizar los renderers cacheados para evitar recorridos de jerarquía
        Renderer[] renderers = preCachedRenderers;
        if (renderers == null || renderers.Length == 0)
        {
            renderers = vehicle.GetComponentsInChildren<Renderer>();
            if (renderers.Length == 0)
            {
                return false;
            }
        }

        // Encapsular Bounds de los Renderers activos
        Bounds bounds = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
        {
            bounds.Encapsulate(renderers[i].bounds);
        }

        // Llenar el buffer estático sin instanciar arreglos nuevos
        FillBoundsCorners(bounds, CornersBuffer);

        float minX = float.PositiveInfinity;
        float minY = float.PositiveInfinity;
        float maxX = float.NegativeInfinity;
        float maxY = float.NegativeInfinity;

        bool objectInFrontOfCamera = false;

        for (int i = 0; i < 8; i++)
        {
            Vector3 viewportPoint = camera.WorldToViewportPoint(CornersBuffer[i]);

            if (viewportPoint.z <= 0)
            {
                continue;
            }

            objectInFrontOfCamera = true;

            if (viewportPoint.x < minX) minX = viewportPoint.x;
            if (viewportPoint.y < minY) minY = viewportPoint.y;
            if (viewportPoint.x > maxX) maxX = viewportPoint.x;
            if (viewportPoint.y > maxY) maxY = viewportPoint.y;
        }

        if (!objectInFrontOfCamera)
        {
            return false;
        }

        // Limitar coordenadas al marco de la imagen (0 a 1)
        minX = Mathf.Clamp01(minX);
        minY = Mathf.Clamp01(minY);
        maxX = Mathf.Clamp01(maxX);
        maxY = Mathf.Clamp01(maxY);

        if (maxX <= minX || maxY <= minY)
        {
            return false;
        }

        box.xMin = minX;
        box.yMin = minY;
        box.xMax = maxX;
        box.yMax = maxY;

        return true;
    }

    private static void FillBoundsCorners(Bounds bounds, Vector3[] buffer)
    {
        Vector3 min = bounds.min;
        Vector3 max = bounds.max;

        buffer[0] = new Vector3(min.x, min.y, min.z);
        buffer[1] = new Vector3(min.x, min.y, max.z);
        buffer[2] = new Vector3(min.x, max.y, min.z);
        buffer[3] = new Vector3(min.x, max.y, max.z);

        buffer[4] = new Vector3(max.x, min.y, min.z);
        buffer[5] = new Vector3(max.x, min.y, max.z);
        buffer[6] = new Vector3(max.x, max.y, min.z);
        buffer[7] = new Vector3(max.x, max.y, max.z);
    }

    public static Rect GetScreenRect(
        BoundingBox box,
        int width,
        int height)
    {
        float x = box.xMin * width;
        float y = box.yMin * height;
        float w = box.Width * width;
        float h = box.Height * height;

        return new Rect(x, y, w, h);
    }

    public static string ToYoloString(
        VehicleType vehicleType,
        BoundingBox box)
    {
        int classId = (int)vehicleType;

        return string.Format(
            CultureInfo.InvariantCulture,
            "{0} {1:F6} {2:F6} {3:F6} {4:F6}",
            classId,
            box.CenterX,
            box.CenterY,
            box.Width,
            box.Height
        );
    }
}