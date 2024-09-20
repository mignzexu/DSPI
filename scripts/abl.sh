savefile="out/inf/abl.log"

# "UCF50" "QNRF" "JHU" "PUCPR" "ShanghaiTech" "FSC_gdino" "CARPK"
for dataset in "FSC_gdino" "CARPK" "ShanghaiTech"; do
    if [[ $dataset == "UCF50" ]]; then
        for id in 1 2 3 4 5 0; do
            echo bash scripts/test.sh -d $dataset -s $id | tee -a $savefile
            bash scripts/test.sh -d $dataset -s $id | tee -a $savefile
            echo bash scripts/test.sh -d $dataset -s $id -b | tee -a $savefile
            bash scripts/test.sh -d $dataset -s $id -b | tee -a $savefile
        done
    elif [[ $dataset == "ShanghaiTech" ]]; then
        for id in "A" "B"; do
            echo bash scripts/test.sh -d $dataset -s $id | tee -a $savefile
            bash scripts/test.sh -d $dataset -s $id | tee -a $savefile
            echo bash scripts/test.sh -d $dataset -s $id -b | tee -a $savefile
            bash scripts/test.sh -d $dataset -s $id -b | tee -a $savefile
        done
    else
        echo bash scripts/test.sh -d $dataset | tee -a $savefile
        bash scripts/test.sh -d $dataset | tee -a $savefile
        echo bash scripts/test.sh -d $dataset -b | tee -a $savefile
        bash scripts/test.sh -d $dataset -b | tee -a $savefile
    fi
done
