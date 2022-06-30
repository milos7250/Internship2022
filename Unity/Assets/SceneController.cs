using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SceneController : MonoBehaviour
{
    private GameObject[] datasets;

    // Start is called before the first frame update
    void Start()
    {
        datasets = GameObject.FindGameObjectsWithTag("Dataset");
        changeScene(0);
    }

    // Update is called once per frame
    void Update()
    {
        
    }




    public void changeScene(int datasetNumber)
    {
        datasetNumber = datasetNumber + 1;

        foreach (GameObject dataset in datasets)
        {
            if (dataset.name.Equals("trimesh_dataset_" + datasetNumber))
            {
                dataset.SetActive(true);
            }
            else
            {
                dataset.SetActive(false);
            }
        }
    }
}
