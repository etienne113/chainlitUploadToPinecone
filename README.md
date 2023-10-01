# chainlitUploadToPinecone
You can use this chainlit interface to upload your documents to your Pinecone index.

## Installation

1- Clone this git repository by running the command: 
```shell
  git clone https://github.com/etienne113/chainlitUploadToPinecone.git
```
  
2- Open the cloned repository in a IDE of your choice, we recommand PyCharm

  Click of this link to download the PyCharm IDE: https://www.jetbrains.com/pycharm/download/?section=windows#section=windows
  
3- Open the terminal in Pycharm: 

  ![image](https://github.com/etienne113/chainlitUploadToPinecone/assets/96786848/7f313354-27f0-4f6e-934c-51815132ea60)
  
4- Download and install  Python (if not already installed) : visit the website https://www.python.org/downloads/

5- run the command:
  ```shell
   python3 -m venv venv
  ```
  and then copy this and paste into your terminal:
  ```shell
  . venv/bin/activate
  ```
  
6- Now install the required dependencies by running the command:
```shell
  pip install -r requirements.txt
```
7- Create a .env file from the .env.example file by running the command:
  ```shell
    cp .env.example .env
  ```
and then fill the empty fields.

8- Now you can run your programm by running the command:
```shell
    . venv/bin/activate
```
and then: 
```shell
    chainlit run document_qa_using_pinecone.py -w
```

  Thank you for the visit! 
