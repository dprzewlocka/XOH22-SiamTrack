**Xilinx Open Hardware 2022**, **Team number: xohw22-113**

# **Real time and energy efficient Siamese tracking**

Link to YouTube Video(s): https://www.youtube.com/watch?v=wfYoSmGbvA8



*University name:* AGH University of Science and Technology in Krakow, Poland

*Participant(s):* Dominika Przewłocka-Rus, e-mail: dprze@agh.edu.pl

*Supervisor:* Tomasz Kryjak, e-mail: tomasz.kryjak@agh.edu.pl

 
## Technicals

Board used: **Zynq UltraScale+ MPSoC ZCU104**

Software Version:
* Vivado 2020.1
* FINN 0.7 (and everything setup for the FINN)
* SD image: v2.6 (https://pynq.readthedocs.io/en/latest/pynq_sd_card.html)


## Brief description:
Siamese trackers have been among the state-of-the-art solutions in each Visual Object Tracking (VOT) challenge over the past few years. However, with great accuracy comes great computational complexity: to achieve real-time processing, these trackers have to be massively parallelised. For this project we propose a hardware-software implementation of the well-known fully connected Siamese tracker (SiamFC). We have developed a quantised Siamese network for the FINN accelerator, using algorithm-accelerator co-design. For our network, running in the programmable logic part of the Zynq UltraScale+ MPSoC ZCU104, we achieved the processing of almost 50 frames-per-second with tracker accuracy on par with its floating point counterpart, as well as the original SiamFC network. The complete tracking system, implemented in ARM with the network accelerated on FPGA, achieves up to 17 fps.

:green_book: Publication (accepted for DASIP 22 conference): https://arxiv.org/abs/2205.10653


## Repository content

 ```
* hardware_acceleration
---> siam_track
    ---> XOH
        ---> data
            ---> Crossing
                ---> img
                    ---> ... (frames from OTB Crossing sequence)
                ---> parameters
                    ---> ... (npy parameters for tracker initialization for chosen sequence)
                groundtruth_rect.txt (groundtruth for Crossing sequence)
            Add_0_param0.npy (parameters to be add to the output of FPGA-executed network - used by Pynq driver)

        ---> driver_backup (files to be used instead of autogenerated by FINN - instructions in the notebook)
            ---> data_packing.py
            ---> driver.py
            ---> driver_base.py

        ---> output/Crossing (empty folder for tracking results - generated with running option 1, see notebook instructions)
        siamfc.pth (trained Siamese network to be used for hardware implementation)

    siam_track_XOH.ipynb (notebook for generating the hardware and testing, with detailed instructions)

* software_tracker
---> siamfc
    ---> contains .py scripts with tracker definition
---> tools
    ---> pretrained (containes pretrained network)
    ---> demo.py (run software baseline quantized tracker for chosen sequence)
    ---> FINN_preparation.py (saves network in FINN-friendly version)
    ---> test.py, train.py (used for evaluation and training)
---> LICENSE (MIT License)
```



## Instructions to build and test project

To run the demo of tracker on ZCU 104 platform you need the hardware_acceleration/siam_track files (the content of software_tracker is additional, so one can further extend this work, if one wishes to do so - this will be discussed at the end of this readme).

#### Step 1:
Copy siam_track folder from hardware_acceleration folder to notebooks directory in your finn installation directory on host.
Run notebooks and open siam_track_XOH.ipynb. The siam_track/XOH directory contains all needed files.

#### Step 2:
Follow the notebook instructions to build the hardware project. You can skip the Hardware Generation section (which takes around 2h to pass) and use the generated design if you wish - all onnx models, as well as all intermediate and final hardware designs, with bitstream and PYNQ driver are to be found in generated_folding_v5.zip archive available here https://drive.google.com/file/d/1EbOCyJM5uoeRIXz3JzMR0PK-TIK-qWGt/view?usp=sharing. But I strongly recommend to generate them yourself to avoid strange problems (although if you're fluent in FINN, or at least have SOME experience, go for it).

Brief summary of instructions from notebook:
* run all cells before Deployment and Execution section (Hardware Generation can take some time - around an hour and a half in my case, so be patient)
* set proper values in first two cells of Deployement and Remote Execution section to match the configuration of your ZCU 104 board
* run cells up to Updating the PYNQ driver
* follow the instructions to update the driver's code
* run cells in Updating the PYNQ driver, part 1 section to move files on board, follow instructions for second driver update
* continue with instructions in What next section of notebook to run the project in hardware

#### Step 3 (just for your consideration):
If you want to test other video sequences, as pointed out in What next section of notebook, you can use the Python project to generate needed initialization parameters. You need to run demo.py script, with the chosen sequence: in the SiamTrack class definition, during initialization of the tracker, several npy files are saved. You need to use them, instead of the attached for Crossing sequence files (you will need to update the paths in Pynq driver and/or the code for option 1).

The Python projects contains also the whole tracker code. It must be noted that the code for Siamese Tracker is based on https://github.com/huanglianghua/siamfc-pytorch implementation.
