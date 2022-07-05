### 目录结构
```shell
.
├── code
│   ├── README.md #This file
│   ├── conll #conll2003 english dataset
│   │   ├── metadata
│   │   ├── test.txt
│   │   ├── train.txt
│   │   └── valid.txt
│   ├── conll2002 #conll2002 dataset
│   │   ├── README
│   │   ├── esp.testa
│   │   ├── esp.testb
│   │   ├── esp.train
│   │   ├── ned.testa
│   │   ├── ned.testb
│   │   └── ned.train
|   ├── requirements.txt
│   ├── dataproc.py
│   ├── embedding.py
│   ├── getglove.sh
│   ├── mainmodel.py
│   └── traintest.py #main
└── doc.pdf
```

## 运行

运行前请先执行`getglove.sh`获取GloVe embedding。

主程序为 `traintest.py`

所有参数在`traintest.py`第98行处，有注释说明。

本程序所有输出均在stdout。通过命令行输入输出重定向到文件即可。
