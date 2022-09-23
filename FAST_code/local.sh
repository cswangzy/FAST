rank=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29")
#ip=("33" "34" "35" "21" "22" "23" "24" "25" "29" "30")
for i in 0 1 2 3 4 5  
do
python device.py --device_num 30 --node_num $i --use_gpu_id 4 --model_type 'LeNet' --dataset_type 'emnist'&
done
for i in 6 7 8 9 10 11 12 13
do
python device.py --device_num 30 --node_num $i --use_gpu_id 5 --model_type 'LeNet' --dataset_type 'emnist'&
done
for i in 14 15 16 17 18 19 20 21
do
python device.py --device_num 30 --node_num $i --use_gpu_id 6 --model_type 'LeNet' --dataset_type 'emnist'&
done
for i in 22 23 24 25 26 27 28 29
do
python device.py --device_num 30 --node_num $i --use_gpu_id 7 --model_type 'LeNet' --dataset_type 'emnist'&
done
# lsof -i:Pid
# kill -9 
#pkill -f "device\.py --device_num*"
#python PS.py --device_num 30 --model_type 'AlexNet' --dataset_type 'cifar10' --alg_type 0
#python PS.py --device_num 30 --model_type 'ResNet' --dataset_type 'cifar100' --alg_type 0
#python edge.py --device_num 30 --model_type 'NIN' --dataset_type 'cifar10'
#python edge1.py --device_num 30 --model_type 'NIN' --dataset_type 'cifar10'
#python PS.py --device_num 1 --model_type 'LeNet' --dataset_type 'emnist' --alg_type 1
#python device.py --device_num 1 --model_type 'LeNet' --dataset_type 'emnist'

# rank=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29")
# #ip=("33" "34" "35" "21" "22" "23" "24" "25" "29" "30")
# for i in 0 1 
# do
# python device.py --device_num 10 --node_num $i --use_gpu_id 0 --model_type 'AlexNet' --dataset_type 'cifar10'&
# done
# for i in 2 3 4
# do
# python device.py --device_num 10 --node_num $i --use_gpu_id 1 --model_type 'AlexNet' --dataset_type 'cifar10'&
# done
# for i in 5 6 7
# do
# python device.py --device_num 10 --node_num $i --use_gpu_id 2 --model_type 'AlexNet' --dataset_type 'cifar10'&
# done
# for i in 8 9
# do
# python device.py --device_num 10 --node_num $i --use_gpu_id 3 --model_type 'AlexNet' --dataset_type 'cifar10'&
# done