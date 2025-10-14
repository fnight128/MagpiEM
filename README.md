# MagpiEM
MagpiEM is a program for automating much of the cleaning process during sub-tomo averaging of proteins in regular lattices, such as membrane proteins or viral coats. 

See the poster [here](https://figshare.com/articles/poster/MagpiEM_Poster/23631759/).

The software is not suitable for proteins which do not form any kind of regular lattice, such as individual ribosomes in solution. 
Please note that the software is still very actively in development, and you may run into errors. Please report any issues here or by email.

## Installation

MagpiEM is currently designed to be run locally on your desktop, rather than on a high-performance computation cluster. 

### Option 1: Containerised Installation

1. Install Docker on your system:
   - Windows / mac: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Linux: [Docker Engine](https://docs.docker.com/engine/install/)

2. Clone the repository to an easily accessible directory and build the container:
   ```bash
   git clone https://github.com/fnight128/MagpiEM.git
   cd MagpiEM
   docker-compose up --build
   ```

3. Open your preferred web browser and connect to `http://localhost:8050`

Next time you wish to run the application, navigate back to the MagpiEM directory and run
   ```bash
   docker-compose up
   ```

Note that if using this method, your browser will not be opened automatically, due to the nature of the container.

### Option 2: Install with pip

Open a python terminal (e.g. [anaconda](https://www.anaconda.com/)), with python 3.10 or above, and run
   ```bash
   pip install MagpiEM
   ```

This should install the software and all dependencies. **The package includes a C++ component that will be automatically compiled during installation.** This process requires a C++ compiler to be available on your system:

- **Windows**: Visual Studio Build Tools or MinGW-w64
- **macOS**: Xcode Command Line Tools (install with `xcode-select --install`)
- **Linux**: GCC or Clang (usually pre-installed)

If you encounter compilation issues, please ensure you have the appropriate development tools installed for your operating system. If you run into issues with this method, containerised installation (option 1) is generally more robust.

To start the software, run:
   ```bash
   magpiem
   ```
This will start the software and connect to it in your default browser.


## Usage

Upload the file containing your particle data.

You may choose to read a small number of tomograms initially, to test the software on a small dataset and find good parameters.

Once you’re happy, choose "read tomograms". The software will read all tomograms within the file.

## Setting Cleaning Parameters
The parameters used for cleaning will vary based on the system. MagpiEM can calculate these parameters directly, for ease of use. Locate a good lattice within the uncleaned tomogram which appears in white. Click two adjacent particles, and the parameters between those will be printed above the diagram.

If your lattice has a large variance in curvatures, you may wish to repeat this for multiple particles, in order to get an idea of the overall range for the whole lattice. These parameters can then be entered into the cleaning form, using the tolerance option to account for variability. Typically, tolerances can be relatively large without much issue(10-20 in each case), but make sure to check on a small number of images first.

### Parameters
| Parameter | Description |
| ----------- | ----------- | 
| Distance      | Distance in pixels between adjacent particles in lattice | 
| Orientation   | Angle between adjacent particles in lattice (specifically, the z-components of their rotation matrices) |
| Curvature   | Triangulated curvature based on the angle between a particle's orientation and the surface of the lattice | 
| Min. Neighbours | Minimum number of adjacent particles in lattice for a particle to be considered valid. Suggested to keep at 2-3| 
| CC Threshold | Minimum CC threshold for particles to be considered, as calculated during template matching within RELION/emClarity. NOTE: currently unreliable with RELION files. |
| Min. Lattice Size | After formation of lattices, any containing less than this many particles will be discarded. |
| Allow Flipped Particles | If enabled, particles with alternating, opposite orientations will be allowed, and count as correct |

## Choosing Correct Lattices
After cleaning has been run, inspect the tomograms given. The option "Show removed particles" can be helpful to ensure no good particles have been removed. 

Once you’re satisfied with the automated step, the last step is to remove any unwanted lattices if necessary. Typically these will be lattices with opposite orientations, which obey the lattice geometry but were picked erroneously during template matching.

Selecting "Cone plot" can be useful here, but due to some very unfortunate quirks of the graphing software, this can sometimes generate comically large cones (see "cone_fix_readme.txt" for more information). Simply clicking "Set" next to "Overall Cone Size" will fix this problem. I hope to one day find a more pleasing solution.

The final step is to select which lattices are correct in the results given. Chosen lattices are highlighted in white. There are two ways to do this, depending on how many correct lattices are present in each image:
*	If only a small number of correct lattices are present in each image (for example, if the image has been divided into sub-regions for template matching), it is more convenient to select each correct lattice. In this case, ensure the toggle under "Save Result" is set to "Keep selected particles"
*	If a large number of correct lattices are present (for example, a picking result of an entire undivided image at once), it may be impractical to select each correct lattice. Instead, select each incorrect lattice, and then change the setting under "Save Result" to "Keep unselected particles". This will save all lattices EXCEPT the ones chosen.

Buttons are available to select all concave or convex lattices i.e. lattices with curvature greater than or less than 90°. This can sometimes prove useful.

## Saving Result
Clicking "Save Particles" will create a file identical to the uploaded file (.mat or .star), with the unwanted particles removed. This can go directly back into emClarity/RELION without any conversion.


## Pausing Manual Selection
If your dataset contains a large number of images, it may be preferable to inspect the lattices over multiple sessions. In this case, click "Save Current Progress" at any time to save a snapshot of your work. This will include all autocleaning steps (so you will not have to set parameters or run autocleaning again), as well as saving the lattices you’ve already selected (so you will not have to click lattices again that you’ve already clicked). This will create a file ending in "progress.yml", which contains all the work you've done so far in a human-readable format.

To load your previous progress, upload the same .mat/.star file as before, but click "Load previous session" instead of "Read tomograms". This may take a few seconds. Do NOT click "Read tomograms" at any point!
 
Note that because this feature stores the results of automated cleaning, it cannot be used if only a subset (1 or 5) of the tomograms were initially loaded - in this case, the software has no data for which particles to include in the rest of the tomograms.
