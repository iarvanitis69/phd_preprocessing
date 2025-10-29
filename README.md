# 📘 DataAcquisition Repository

This repository contains the code for data acqisition.

---

## 🧩 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://gitlab.com/iarv/phd.git
cd phd
```

### 2️⃣ Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
# .\venv\Scripts\activate  # On Windows
```

### 3️⃣ Install Dependencies for a Specific Subproject
Each subproject has its own `requirements.txt` file. Before running a project, install its dependencies separately.

**Example – DataAcquisition:**
```bash
cd DataAcquisition
pip install -r requirements.txt
```
---

## 🚀 Execution

### Run the DataAcquisition Project
```bash
cd DataAcquisition
python mainEventBase.py <YEAR>
```

**Example:**
```bash
python mainEventBase.py 2024
```

---



