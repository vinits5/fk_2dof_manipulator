# Forward Kinematics of 2DOF Manipulator using Neural Network

This repository contains a nueral network trained to predict the end effector position of a 2DOF robotic arm given its joint positions.

### Codes:
**fk_2dof_tensorflow.py** file has the training algorithm.\
**neural_network** directory has the network architecture.\
**log_data** contains trained weights of network.
**fk_data.py** can be used to generate new data to train the network.

### Data:
**fk_data_ip.csv** contains the input angles for network.\
**fk_data_op.csv** contains the output or predicted end-effector position using robot kinematics.

## Results

### Simulation
<p align="center">
	<img src="https://github.com/vinits5/fk_2dof_manipulator/blob/master/robot_simulation/fk_2dof_neural_net.gif" title="Robot Simulation">
</p>

### Loss Function
<p align="center">
	<img src="https://github.com/vinits5/fk_2dof_manipulator/blob/master/log_data/2018-08-20-01-29-59/Loss_per_step.png" title="Loss Function">
	<img src="https://github.com/vinits5/fk_2dof_manipulator/blob/master/log_data/2018-08-20-01-29-59/results.png" title="Results">
</p>
