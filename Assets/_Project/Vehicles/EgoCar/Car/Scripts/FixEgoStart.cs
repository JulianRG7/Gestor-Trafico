using UnityEngine;

public class FixEgoStart : MonoBehaviour
{
    void Start()
    {
        // Forzamos la posición inicial al primer frame según tus necesidades (-145 en Z)
        transform.position = new Vector3(0f, 0.5f, -145f);
        transform.rotation = Quaternion.Euler(0, 0, 0);

        // Bloqueamos la física para que no se caiga ni se mueva solo
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.isKinematic = true;
            rb.useGravity = false;
        }
    }
}