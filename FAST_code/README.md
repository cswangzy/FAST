
This is the code for Heal based on pytorch, which contains predefined models on popular dataset. Currently we support

* model: AlexNet, VGG and NIN
* Dataset: Emnist, Cifar and ImageNet

# Running 
You should initilize PS, and device in order. For example, if we want to run a system with one PS, and device for training NIN over Cifar10,

* python PS.py --device_num 1 --edge_number 1 --model_type 'NIN' --dataset_type 'cifar10' 
* python device.py --device_num 1 --node_num 1 --use_gpu 0 --model_type 'NIN' --dataset_type 'cifar10'

You can change those settings including device_num, model_type and dataset_type. By using "--model_type 'AlexNet' --dataset_type 'Emnist' ", you can train alexnet over emnist.

# Settings

* Ip: If you want to run code locally, you should set the ip as 'localhost', and assign different ports for different workers. "listening_sock.bind(('localhost', 50010))". Otherwise, if you run code on different equipments, you should assign the real ip to create the connection. 
* Dataset: To run the code safely, you should modify the file path of each dataset as the real path in your computer.




