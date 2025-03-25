**apt update**
```bash
sudo apt udpate
```

**unzip**
```bash
sudo apt install unzip
```

**git**
```bash
sudo apt install git
```

**clone**
```bash
git clone https://github.com/rogervendrell/gnn_pathplanning.git
```

**pip3**
```bash
sudo apt install python3-pip
```

**venv**
```bash
sudo apt install python3-venv
```

**myenv**
```bash
python3 -m venv myenv && source myenv/bin/activate
```

**requirements**
```bash
pip3 install -r gnn_pathplanning/requirements.txt
```

**fix matplotlib**
```bash
pip3 uninstall matplotlib && pip3 install matplotlib==3.1.2
```

**results dir**
```bash
mkdir trainingResults trainingResults/Tensorboard_CG50
```

**torchsummary**
```bash
pip3 install torchsummaryX
```

**hashids**
```bash
pip3 install hashids
```

**rename files**
```bash
mv gnn_pathplanning/graphs/models/decentralplanner.py gnn_pathplanning/graphs/models/decentralplanner_GAT.py && mv gnn_pathplanning/graphs/models/decentralplanner_normal.py gnn_pathplanning/graphs/models/decentralplanner.py
```

**train**
```bash
cd gnn_pathplanning/ && python3 main.py configs/dcp_ECBS_GAT_gcloud.json --mode train  --map_w 20 --nGraphFilterTaps 3  --num_agents 10  --trained_num_agents 10
```