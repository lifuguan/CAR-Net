python unetzoo/main.py -m design_one  -d liver -l hybrid -g 1 --theta 0.0005
python unetzoo/main.py -m design_one  -d liver -l BCE -g 1 --theta 0.0005

python unetzoo/main.py -m design_one  -d lung -l hybrid -g 1 --theta 0.0005
python unetzoo/main.py -m design_one  -d lung -l BCE -g 1 --theta 0.0005

python unetzoo/main.py -m design_one  -d isbicell -l hybrid -g 1 --theta 0.0005
python unetzoo/main.py -m design_one  -d isbicell -l BCE -g 1 --theta 0.0005

python unetzoo/main.py -m design_one  -d dsb2018Cell -l hybrid -g 1 --theta 0.0005
python unetzoo/main.py -m design_one  -d dsb2018Cell -l BCE -g 1 --theta 0.0005

python unetzoo/main.py -m design_one  -d driveEye -l hybrid -g 1 --theta 0.0005
python unetzoo/main.py -m design_one  -d driveEye -l BCE -g 1 --theta 0.0005

python unetzoo/main.py -m UNet  -d lung -l hybrid -g 1 --theta 0.0005
python unetzoo/main.py -m unet++  -d lung -l hybrid -g 1 --theta 0.0005
python unetzoo/main.py -m Attention_UNet  -d lung -l hybrid -g 1 --theta 0.0005
python unetzoo/main.py -m cenet  -d lung -l hybrid -g 1 --theta 0.0005
python unetzoo/main.py -m fcn32s  -d lung -l hybrid -g 1 --theta 0.0005

python unetzoo/main.py -m UNet  -d lung -l BCE -g 1 --theta 0.0005
python unetzoo/main.py -m unet++  -d lung -l BCE -g 1 --theta 0.0005
python unetzoo/main.py -m Attention_UNet  -d lung -l BCE -g 1 --theta 0.0005
python unetzoo/main.py -m cenet  -d lung -l BCE -g 1 --theta 0.0005
python unetzoo/main.py -m fcn32s  -d lung -l BCE -g 1 --theta 0.0005

# python unetzoo/main.py -m design_one  -d isbicell -l BCE -g 1 --theta 0.0005
# python unetzoo/main.py -m design_one  -d dsb2018Cell -l BCE -g 1 --theta 0.0005

# python unetzoo/main.py -m design_one  -d COVID19 -l ACELoss -g 1
# python unetzoo/main.py -m UNet  -d COVID19 -l ACELoss -g 1
# python unetzoo/main.py -m Lite_RASPP  -d COVID19 -l ACELoss -g 1
# python unetzoo/main.py -m unet++  -d COVID19 -l ACELoss -g 1
# python unetzoo/main.py -m Attention_UNet  -d COVID19 -l ACELoss -g 1
# python unetzoo/main.py -m cenet  -d COVID19 -l ACELoss -g 1

# python unetzoo/main.py -m design_one  -d COVID19 -l BCE -g 1
# python unetzoo/main.py -m UNet  -d COVID19 -l BCE -g 1
# python unetzoo/main.py -m Lite_RASPP  -d COVID19 -l BCE -g 1
# python unetzoo/main.py -m unet++  -d COVID19 -l BCE -g 1
# python unetzoo/main.py -m Attention_UNet  -d COVID19 -l BCE -g 1
# python unetzoo/main.py -m cenet  -d COVID19 -l BCE -g 1

# python unetzoo/main.py -m Lite_RASPP        -d liver
# python unetzoo/main.py -m Lite_RASPP        -d isbicell
# python unetzoo/main.py -m Lite_RASPP        -d esophagus
# python unetzoo/main.py -m Lite_RASPP        -d driveEye
# python unetzoo/main.py -m Lite_RASPP        -d dsb2018Cell



# python unetzoo/main.py -m UNet        -d corneal
# python unetzoo/main.py -m resnet34_unet  -d corneal
# python unetzoo/main.py -m unet++ -d corneal
# python unetzoo/main.py -m Attention_UNet -d corneal
# python unetzoo/main.py -m segnet -d corneal
# python unetzoo/main.py -m r2unet -d corneal
# python unetzoo/main.py -m fcn32s -d corneal
# python unetzoo/main.py -m myChannelUnet -d corneal
# python unetzoo/main.py -m cenet -d corneal
# python unetzoo/main.py -m smaatunet -d corneal
# python unetzoo/main.py -m self_attention_unet -d corneal

# python unetzoo/main.py -m UNet        -d esophagus
# python unetzoo/main.py -m resnet34_unet  -d esophagus
# python unetzoo/main.py -m unet++ -d esophagus
# python unetzoo/main.py -m Attention_UNet -d esophagus
# python unetzoo/main.py -m segnet -d esophagus
# python unetzoo/main.py -m r2unet -d esophagus
# python unetzoo/main.py -m fcn32s -d esophagus
# python unetzoo/main.py -m myChannelUnet -d esophagus
# python unetzoo/main.py -m cenet -d esophagus
# python unetzoo/main.py -m smaatunet -d esophagus
# python unetzoo/main.py -m self_attention_unet -d esophagus



# python unetzoo/main.py -m UNet        -d kagglelung
# python unetzoo/main.py -m resnet34_unet  -d kagglelung
# python unetzoo/main.py -m unet++ -d kagglelung
# python unetzoo/main.py -m Attention_UNet -d kagglelung
# python unetzoo/main.py -m segnet -d kagglelung
# python unetzoo/main.py -m r2unet -d kagglelung
# python unetzoo/main.py -m fcn32s -d kagglelung
# python unetzoo/main.py -m myChannelUnet -d kagglelung
# python unetzoo/main.py -m cenet -d kagglelung
# python unetzoo/main.py -m smaatunet -d kagglelung
# python unetzoo/main.py -m self_attention_unet -d kagglelung

# python unetzoo/main.py -m UNet        -d driveEye
# python unetzoo/main.py -m resnet34_unet  -d driveEye
# python unetzoo/main.py -m unet++ -d driveEye
# python unetzoo/main.py -m Attention_UNet -d driveEye
# python unetzoo/main.py -m segnet -d driveEye
# python unetzoo/main.py -m r2unet -d driveEye
# python unetzoo/main.py -m fcn32s -d driveEye
# python unetzoo/main.py -m myChannelUnet -d driveEye
# python unetzoo/main.py -m cenet -d driveEye
# python unetzoo/main.py -m smaatunet -d driveEye
# python unetzoo/main.py -m self_attention_unet -d driveEye

# python unetzoo/main.py -m UNet        -d dsb2018Cell
# python unetzoo/main.py -m resnet34_unet  -d dsb2018Cell
# python unetzoo/main.py -m unet++ -d dsb2018Cell
# python unetzoo/main.py -m Attention_UNet -d dsb2018Cell
# python unetzoo/main.py -m segnet -d dsb2018Cell
# python unetzoo/main.py -m r2unet -d dsb2018Cell
# python unetzoo/main.py -m fcn32s -d dsb2018Cell
# python unetzoo/main.py -m myChannelUnet -d dsb2018Cell
# python unetzoo/main.py -m cenet -d dsb2018Cell
# python unetzoo/main.py -m smaatunet -d dsb2018Cell
# python unetzoo/main.py -m self_attention_unet -d dsb2018Cell

# python unetzoo/main.py -m UNet        -d liver
# python unetzoo/main.py -m resnet34_unet  -d liver
# python unetzoo/main.py -m unet++ -d liver
# python unetzoo/main.py -m Attention_UNet -d liver
# python unetzoo/main.py -m segnet -d liver
# python unetzoo/main.py -m r2unet -d liver
# python unetzoo/main.py -m fcn32s -d liver
# python unetzoo/main.py -m myChannelUnet -d liver
# python unetzoo/main.py -m cenet -d liver
# python unetzoo/main.py -m smaatunet -d liver
# python unetzoo/main.py -m self_attention_unet -d liver

# python unetzoo/main.py -m UNet        -d isbicell
# python unetzoo/main.py -m resnet34_unet  -d isbicell
# python unetzoo/main.py -m unet++ -d isbicell
# python unetzoo/main.py -m Attention_UNet -d isbicell
# python unetzoo/main.py -m segnet -d isbicell
# python unetzoo/main.py -m r2unet -d isbicell
# python unetzoo/main.py -m fcn32s -d isbicell
# python unetzoo/main.py -m myChannelUnet -d isbicell
# python unetzoo/main.py -m cenet -d isbicell
# python unetzoo/main.py -m smaatunet -d isbicell
# python unetzoo/main.py -m self_attention_unet -d isbicell

