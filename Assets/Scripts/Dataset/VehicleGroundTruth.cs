using System.Collections.Generic;
using UnityEngine;

public class VehicleGroundTruth : MonoBehaviour
{
    [Header("Vehicle Information")]
    public VehicleType vehicleType;

    [Tooltip("Optional unique ID for this vehicle.")]
    public int vehicleId;

    [Header("Bounding Box")]
    [Tooltip("If enabled, child Renderers are included in the bounding box.")]
    public bool includeChildren = true;

    // Cache de renderers para no llamar a GetComponentsInChildren repetidamente
    [HideInInspector]
    public Renderer[] CachedRenderers { get; private set; }

    // Registro estático global para evitar FindObjectsByType O(N)
    public static readonly List<VehicleGroundTruth> AllVehicles = new List<VehicleGroundTruth>();

    private void Awake()
    {
        CacheRenderers();
    }

    private void OnEnable()
    {
        if (!AllVehicles.Contains(this))
        {
            AllVehicles.Add(this);
        }
    }

    private void OnDisable()
    {
        AllVehicles.Remove(this);
    }

    public void CacheRenderers()
    {
        CachedRenderers = includeChildren
            ? GetComponentsInChildren<Renderer>()
            : GetComponents<Renderer>();
    }
}