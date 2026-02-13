# Analytics Catalogue

The Analytics Catalogue provides a unified interface for specialized services and libraries developed as part of Work Package 3 (WP3). These tools provide advanced analytics capabilities designed to enhance the machine learning lifecycle, from data selection to model adaptation. WP3 tasks are implemented using two distinct modes based on their integration requirements:

- Services Mode: Independent services instantiated via a common API. These are defined under the /services directory and follow a specific workflow specification.

- Library Mode: Modular components to complement the analytical workflow.


## Services 

| Service | Description | Path |
| :--- | :--- | :--- |
| **Automated Dataset Selection** | Recommends optimized datasets for ML tasks and automated feature augmentation. | `services/data_selection` |
| **Analysis-aware Data Integration** | Supervised entity resolution using Language Models (LM) and GPU acceleration. | `services/data_integration` |
| **Simulation-driven Augmentation** | Generates synthetic data to improve ML model accuracy. | `services/data_augmentation` |s


## Libraries 

| Library | Description | Path |
| :--- | :--- | :--- |
| **Continual ML** | Implements continual learning strategies to adapt models over time. | `libraries/continual_ml` |
| **Opportunistic ML** | Constraint-aware ML models (e.g., performance, topology) designed to optimize learning under dynamic system conditions.. | `libraries/opportunistic_ml` |
| **Meta-Learning** |Implements meta-learning techniques that enable models to learn how to learn, improving fast adaptation to new tasks or evolving data distributions. | `libraries/metalearning_ml` |

​
> **Note:** Each service and library contains a dedicated `README.md` with specific configuration instructions and experimentation examples.
​

## Getting Started 

Before running the project, make sure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) 

In the `config.ini`, modify the host field to match the location where Docker Compose is deployed. As future work, this step could be automated by parsing the output of the `ip` command.


### Run the Docker Compose

To start the project and deploy all services, run the following command:

```bash
docker compose up
```

This command will launch all services defined in the docker-compose.yml file. Once running, you can access the interface at:

```bash
http://0.0.0.0:9000
```

