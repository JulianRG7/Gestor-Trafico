using System.Collections;
using UnityEngine;

public class EmergencyLightsFlasher : MonoBehaviour
{
    [Header("Luces")]
    public Light[] redLights;
    public Light[] blueLights;

    [Header("Ajustes")]
    public float flashInterval = 0.25f;
    public bool lightsActive = true;

    private void Start()
    {
        StartCoroutine(FlashRoutine());
    }

    private IEnumerator FlashRoutine()
    {
        while (true)
        {
            if (lightsActive)
            {
                ToggleLights(redLights, true);
                ToggleLights(blueLights, false);
                yield return new WaitForSeconds(flashInterval);

                ToggleLights(redLights, false);
                ToggleLights(blueLights, true);
                yield return new WaitForSeconds(flashInterval);
            }
            else
            {
                ToggleLights(redLights, false);
                ToggleLights(blueLights, false);
                yield return new WaitForSeconds(0.5f);
            }
        }
    }

    private void ToggleLights(Light[] lights, bool state)
    {
        foreach (Light l in lights)
        {
            if (l != null) l.enabled = state;
        }
    }
}