# BU, ExtremeXP: Meta-learning Search Engine
Application repository for the proof-of-concept Meta-learning Search Engine developed as a small part of the wider Bournemouth University contribution to the ExtremeXP project.
(https://www.bournemouth.ac.uk/research/projects/extremexp).

<br>

## Implementation Overview
This tool aims to provide a pipeline where users compare algorithm performance on different datasets, prioritised according to statistical measures of similarity, thereby analysing which algorithms are more likely to be effective for an individual use case.
<br>

![image](https://github.com/user-attachments/assets/bc999c48-470f-46ac-9fd1-b4ed9227feb2)

<br>

Users upload & compare different datasets. Datasets are split into:
- 1. **‘Reference Dataset’ (i.e. ‘independent’ variable)** - A singular Dataset which requires an algorithm 'reccomendation'
- 2. **‘Comparison Datasets’ (i.e. ‘dependent’ variables)** - Multiple Datasets for the ‘Reference Dataset’ to be compared against (calculate similairty between them)
- 3. **‘Performance Metrics’ (i.e. algorithm statistics)** - For each ‘Comparison Dataset’ containing the performance information of the algorithms that were run on the Dataset (used for algorithm 'reccomendation')

<br>

There are two data upload mechanisms:
- **Interface upload** - All Datasets uploaded to be used in the comparison are stored temporarily in session memory.
- **Database upload** - Comparison datasets can be uploaded to a local MySQL Database for use in subsequent comparison operation with uloaded Reference Datasets

<br>

A ‘task type’ can be set to filter uploaded algorithm performance according to chosen type (Classification/ Regression)

<br>
<br>

## Setting up and running the application for manual reccomendation
These sets of instructions will allow the tool to perform manual reccomendation / comparison. This will require the upload of all the Datasets manually to perform the reccomendation / comparison operation

### 1. Create a Python Virtual Environment (venv)
In the Command Prompt, run:
  - Navigate to an appropriate directory for a venv folder to be created.
  - Run: `python -m venv venv_name` (replacing venv_name).
<br>


### 2. Activate venv
In the Command Prompt:
  - Make sure you are still in the same directory as in Step 1.
  - Run: `venv_name\Scripts\activate`
<br>


### 3. Install dependencies into venv
In the Command Prompt:
  - Navigate to the top-level of the repository (.../metalearning)
  - Run: `pip install -r requirements.txt` to install required dependencies.
<br>


### 4. Deploy/run a local version of the application
Make sure your venv is currently/still activated before progressing.
In the Command Prompt:
  - Navigate to the top-level of the repository (.../metalearning).
  - Run: `streamlit run main.py` to launch the application.
<br>



## Setting up and running the application for Database reccomendation
These sets of instructions will run the tool with a connected Database, allowing the performing of both manual and Database backed reccomendation reccomendation / comparison. Database reccomendation / comparison will still require the upload all of the Reference Dataset manually to perform the comparison operation. The Comparison Datasets and their Performance Metrics Datasets are stored in the Database.
<br>
The prevous steps are still required in order to run the tool. These steps will facilitate the connection of local Database to the tool.

### 1. Install MySQL with MySQL Workbench
Install at:
  - [MySQL + MySQL Workbench download](https://dev.mysql.com/downloads/installer/)
  - Install the appropriate version of MySQL (with MySQL Workbench) for the machine
  - MySQL Workbench will be an optinal installation when installing MySQL
<br>


### 2. Make sure the MySQL server is running
To run the MySQL Server run the command line as admin and perform the following operations:
  - Navigate to MySQL bin directory (where MySQL is installed): `cd "C:\Program Files\MySQL\MySQL Server 8.0\bin"`
  - Start MySQL Server: `mysqld`
<br>


### 3. Create MySQL Connection
Run up MySQL or MySQL WorkBench and make a connection with the following details:
  - Host: localhost
  - User: root
  - Password: Joshua100x
  - Port: 3306

Creating the connection should look something like this (with the password put it):
![image](https://github.com/user-attachments/assets/1f8178f0-ade2-4cee-8753-ab79aa3a869c)

<br>


### 4. Change connection details (only if neccessary)
If the connection details provided in the previous step are inconveniant then they can be changed:
  - Navigate to the top of the Metalearning project files (.../metalearning)
  - Locate the `database_connection.py` file
  - Change the details inside all the `mysql.connector.connect()` objects in the file to the prefered

The connection objects inside the python file should look like this:
![image](https://github.com/user-attachments/assets/3a6dddad-7df8-4f1c-ae48-b7beb8efd63d)

<br>


### 5. Create Database
Create the Database used by the Metalearning search Engine:
  - Create a new query inside of MySQL Workbench
  - Paste and run the following SQL statements in the MySQL Workbench query file
<br>

```
CREATE DATABASE IF NOT EXISTS pca;

USE pca;

DROP TABLE IF EXISTS datasets;
CREATE TABLE datasets (
    dataset_ID int NOT NULL AUTO_INCREMENT,
    dataset_name varchar(50) NOT NULL,
    dataset_data JSON,
    dataset_algorithms JSON,
    CONSTRAINT PK_dataset PRIMARY KEY (dataset_ID)
);
```

<br>
Creating the Database inside MySQL Workbench should look something like this:

![image](https://github.com/user-attachments/assets/47b2d63f-79c5-4575-bce5-467411bbf2d4)


<br>
