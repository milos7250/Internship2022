using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class DatasetController : MonoBehaviour
{
    // The dropdown selecting datasets
    public TMP_Dropdown dropdown;

    // All datasets available have to have the "Dataset" tag to be handled by the dropdown selector
    private GameObject[] datasets;

    // Start is called before the first frame update
    void Start()
    {
        datasets = GameObject.FindGameObjectsWithTag("Dataset");
        changeDataset(0);
    }

    // Update is called once per frame
    void Update()
    {
        
    }


    private string datasetNameNumber;

    public void changeDataset(int datasetNumber)
    {
        datasetNameNumber = dropdown.captionText.text.Substring(8);

        foreach (GameObject dataset in datasets)
        {
            if (dataset.name.Equals("trimesh_dataset_" + datasetNameNumber))
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
