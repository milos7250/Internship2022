# Creating contour visualisations in Unity

This section explains the process of creating a 3D model of a contour from a triangular mesh.

## Python
To be able to use the triangular mesh generated in python in Unity, you need to export the mesh in a suitable file format. There are many formats supported by Unity. I have found that using the `trimesh` Python module to export the mesh in `.glb` format works quite well.

However, when exporting any mesh, I recommend swapping the *y* and *z* axes order to match Unity's coordinate system, i.e. using *y* axis as the *up* axis. In my case, this helped to remove problems with incorrect orientation and inverted normal vectors when importing the mesh into Unity. Swapping the axes also requires to reflect the *z* axis in order for the coordinate systems to match.

An example of how a triangular mesh might be created and imported in Python using the `trimesh` module:
``` python
vertices, faces, normals, _ = skimage.measure.marching_cubes(data)
vertices = vertices[:, (0, 2, 1)]
normals = normals[:, (0, 2, 1)]
vertices[:, 2] *= -1
normals[:, 2] *= -1
mesh = Trimesh(
    vertices=vertices,
    faces=faces,
    vertex_normals=normals,
)
mesh.export("mesh.glb")

```

## Unity
The Unity project already has the camera, background and control elements configured. To add new triangular meshes into the project, you'll need to follow these instructions:

1. Import the triangular mesh by going to **Assets > Import New Asset...**.
2. Drag the newly imported asset from the **Assets** pane into the root of the scene in the **Hierarchy** pane. Rename the game object to `trimesh_dataset_#`, replacing `#` with the number you wish to assign to the dataset.
3. Change the asset's root game object tag from **Untagged** to **Dataset** in the **Inspector** pane.
4. Change the asset's scale in the **Inspector** pane so that it fits neatly into the camera view.
5. Copy the **Sun** game object from any other dataset in the project and set it's parent to be the root of the newly imported asset. Setting the parent can be done by dragging one object onto the other in the **Hierarchy** pane.
6. Find the triangular mesh object of the imported asset and in the **Inspector** pane, set it's material to **Force Field**. Depending on the format of the imported asset, the mesh object might be a child of several other objects within the imported asset's tree.
7. In the **Hierarchy** pane, select **Controls/Dropdown**. In the **Inspector** pane of the Dropdown object, scroll down to the options list and add an entry by clicking on the plus sign. The text of the entry should read `Dataset #`, where `#` is the same number as in step *2.*

Following these steps will make sure that the dropdown menu in the bottom left corner properly shows and hides requested datasets and set's all visual properties to match all other meshes' look.

Alternatively, you might follow the instructions in this GIF:

![A gif guide](./Unity%20Guide.gif)

## Creating images and videos from Unity
If you wish to create images or videos from the unity renderings, you have two choices. You can either build the Unity app (explained in the next section), run the app and record the screen using any screen recording software, or use the Unity's built-in recorder tool.

The recordings can be created by going to **Window > General > Recorder > Recorder Window** and setting the required recording parameters. With the default camera setting that rotates around the Sun on a circular path, the whole rotation takes 18 seconds.

The camera's path can be changed in the script `Assets/CameraRotate.cs`.

By default, the first dataset is loaded when making the recording. To change that, either use the dropdown menu to change the dataset in the beginning, or move the option corresponding to the wanted dataset to the first place in the dropdown's game object inspector pane.

## Building the Unity app
To create a standalone app that can be run to show the animations without the need to install Unity, go to **File > Build Settings...** The settings in this project are good default choices, only the platform should be adjusted to Windows, Linux, etc.
