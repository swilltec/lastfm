# LastFM Asia Social Network Analysis

This project presents a structural and algorithmic analysis of the LastFM Asia Social Network — an undirected graph with 7,624 nodes and 27,806 edges. Each node represents a LastFM user from an Asian country, with features based on the artists they follow and labels indicating their geographic location.

Using Python libraries such as NetworkX and Matplotlib, the analysis includes:

- Graph representation using an adjacency list.
- Detection of cycles (triangles, quadrilaterals, and more).
- Connectivity checks and path existence.
- BFS and DFS graph traversals.
- Visualization of subgraphs.

---

## 📁 Project Structure

```
lastfm/
├── src/
│   └── main.py               # Entry point for the analysis
├── edges.csv                 # Edgelist file (format: node1,node2)
├── requirements.txt          # Required Python packages
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/swilltec/lastfm.git
cd lastfm
```

### 2. Create a Virtual Environment

Make sure you have Python 3.8 or later installed.

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Analysis

```bash
python src/main.py
```

---


## 📚 Dataset Citation

Rozemberczki, B., et al. (2020).  
*LastFM Asia Social Network Dataset*.  
[GitHub Repository](https://github.com/benedekrozemberczki/FEATHER)

---

## 📝 License

This project is intended for educational and research purposes only.
