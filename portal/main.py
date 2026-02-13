import streamlit as st
import pandas as pd
import numpy as np

from typing import Dict, Any
import socket
import requests
import yaml
import configparser


def parse_docker_compose_services(path_to_docker_compose: str = "../docker-compose.yaml") -> Dict[str, Dict[str, Any]]:
    """
    Parse the docker-compose.yaml file to extract service information, including labels.
    
    Args:
        path_to_docker_compose (str): Path to the docker-compose.yaml file.
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary containing service information, including labels.
    """
    with open(path_to_docker_compose, 'r', encoding='utf-8') as file:
        compose_data = yaml.safe_load(file)

    services_info: Dict[str, Dict[str, Any]] = {}

    services = compose_data.get("services", {})
    for service_name, service_data in services.items():
        services_info[service_name] = {
            "container_name": service_data.get("container_name"),
            "ports": service_data.get("ports", []),
            "networks": service_data.get("networks", []),
            "labels": service_data.get("labels", [])  # Add the labels to the dictionary
        }

    return services_info


def ping_service(service_url: str) -> bool:
    """
    Ping a service to check if it's running.
    Args:
        service_name (str): The name of the service to ping.
    Returns:
        bool: True if the service is running, False otherwise.
    """
    host, port = service_url.split(":")
    port = int(port)
    # Create a socket connection to the service
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.settimeout(1)
        return True
    
    except socket.error as e:
        return False
    
    except Exception as e:
        return False
      
def check_api_status(service_url: str) -> bool:
    """
    Check the API status of a service.
    Args:
        service_url (str): The URL of the service to check.
    Returns:
        bool: True if the API is reachable, False otherwise.
    """
    try:
        service_url = f"http://{service_url}/docs"
        response = requests.get(service_url, timeout=2)
        return response.status_code == 200
    
    except requests.RequestException:
        return False


def prettify_name(name: str) -> str:
    """
    Prettify a string for display.
    Args:
        name (str): The original string.
    Returns:
        str: The prettified string.
    """
    return name.replace("_", " ").capitalize()


def render_webapp(host: str, services_dict_info: Dict[str, Dict[str, Any]]) -> None: 
    """
    Render the webapp using Streamlit.
    Args:
        services_dict_info (Dict[str, Dict[str, Any]]): Dictionary containing service information.
    """
    st.set_page_config(page_title="Catalogue Analytics", page_icon=":guardsman:", layout="wide")

    st.title("Catalogue Analytics")
    st.subheader("Service Status:")
    
    for service, info in services_dict_info.items():

        if service== "analytics_catalogue": 
            continue
        
        ports = info.get("ports", [])
        labels = info.get("labels", [])
        labels_dict = {label.split("=")[0]: label.split("=")[1] for label in labels}

        if ports:
            port_mapping = ports[0]
            try:
                host_port = port_mapping.split(":")[0]
                url = f"{host}:{host_port}"
                status_server, status_api = ping_service(url),check_api_status(url)

            except IndexError:
                url = "unknown"
                status = False
        else:
            url = "unknown"
            status = False

        with st.container(border=True):
            st.markdown(f"### 🔹 {prettify_name(service)} ")
            st.markdown("---")
            st.markdown(f"**Server Running:** {'🟢 Running' if status_server else '🔴 Down'}")
            st.markdown(f"**API Status:**     {'🟢 Running' if status_api else '🔴 Down'}")
            
            if labels_dict["is_api_service"] == "true": 
                st.markdown(f"**URL:** `{url}` **API Reference**: [API Documentation](http://{url}/docs)")
            else:
                if "exposed_service" in labels_dict:
                    st.markdown(f"**Exposed services**: `{url}` [URL](http://{url})")
           


if __name__ == "__main__": 

    config = configparser.ConfigParser()
    config.read('config.ini')

    host = config.get("DEFAULT", "host")
    path_to_docker_compose = config.get("DEFAULT", "path_to_docker_compose")
    
    services_dict_info = parse_docker_compose_services(path_to_docker_compose) 
    render_webapp(host, services_dict_info)