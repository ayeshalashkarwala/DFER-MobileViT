import os
for i in range(10):
     # cmd = 'python3 combinemodelkmu.py --model Ourmodel --bs 128 --lr 0.001 --fold %d' %(i+1)
     cmd = 'python3 combinemodelkmu.py --model gcvit --bs 128 --lr 0.001 --fold %d' %(i+1)
     # cmd = 'python3 combinemodelkmu.py --model mobilevit --bs 128 --lr 0.001 --fold %d' 

     os.system(cmd)
print("Train Mobile-ViT ok!")
os.system('pause')