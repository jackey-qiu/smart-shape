# smart-shape
Tool to build complicated objects from basic shapes using qpainter device (QT).

Using smart-shape, you can easily create a drawing containing many objects like this:

![example drawing!](.//assets//imgs//optics_table.png 'optics table')

As you can see, this view is composed of many composite object, which contains numerous basic shapes in specific alignment patterns. The basic shapes include rectangle (scquared or rounded edge), triangle, circle, trapezium. Each basic constitual shape can be decorated with different filling pattern, borderline pattern (color and linestyle). Text labeling is possible for each shape, and the text location and style can be specified. In addition, line connection between shapes are also possible. 

The real powerful thing in the shape viewer is that each shape is clickable and the style will change once the cursor is on the shape. The action of mouseclick can be specified as callback functions. The motivation of this project is driven by the need of a centralized monitor with pitorial style for monitoring a synchrotron beamline status. Like the example drawing shown above, it is a viewer monitoring our equipments on one shared-optics table. Each component is connected to a device model for live state updating. The communication protocal is through taurus and blissclient. 

A simple view of the logic flow of smart-shape is as follows:

![logic flow chart!](.//assets//imgs//logic_flow.png 'logic flowchart')

# Installation guide
1. install pdm Python package and dependency manager following [pdm installation](https://pdm-project.org/en/latest/#installation).
2. clone this project to your local pc using: `git clone https://github.com/jackey-qiu/smart-shape.git`.
3. cd to smart-shape: `cd smart-shape`
4. install the package using pdm: `pdm install`
5. Once through the installation, you can activate the venv using `pdm venv activate in-project`, which will return a link the the venv activation script. Just copy the script and paste is in the terminal to finally activate the venv.
6. In the venv, you can simply type `smart_shape` to launch the GUI. You will see the main gui like this

![smart-shape maingui!](.//assets//imgs//main_gui.png 'optics table')

You can now choose the viewer file and the viewer objects to render viewer. You will probably need to make your own shape file in yaml format.  You should place the file into src/smart_shape/res/shape_meta_data folder to be seen by smart gui.

Two example files are given. The one named viewer_model_free.yaml is shape meta info without using any communication model. The other one named viewer_model.yaml is one based on taurus and blissclient communication model. If you have no idea what are taurus and blissclient, you should try the model free viewer file. 

# shape alignment patterns
In yaml file, you can specify how two shapes are aligned to each other. The possible alignment anchor keys include 'top', 'bottom', 'left', 'right', 'cen' and the names of all anchor attributes. Besides alignment key, you need also reference anchor (geometry center if not specified) to make the alignment. These two parameters (alignment and reference anchor) define the vector, along which two shapes are aligned to each other. 
See some examples here

![alignment pattern!](.//assets//imgs//shape_alignments.png 'alignment patterns')