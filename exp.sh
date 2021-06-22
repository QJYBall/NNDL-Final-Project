nohup python main.py -nexp=baseline -d=4 &
nohup python main.py --mixup -nexp=mixup -d=5 &
nohup python main.py --cutout -nexp=cutout -d=6 &
nohup python main.py --cutmix -nexp=cutmix -d=7 &

nohup python main.py --network=resnet101 --dataset=cifar100 -nexp=baseline -d=4 &
nohup python main.py --network=resnet101 --dataset=cifar100 --mixup -nexp=mixup -d=5 &
nohup python main.py --network=resnet101 --dataset=cifar100 --cutout -nexp=cutout -d=6 &
nohup python main.py --network=resnet101 --dataset=cifar100 --cutmix -nexp=cutmix -d=7 &
