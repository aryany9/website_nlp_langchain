import requests
from termcolor import colored

from cli.spinner import spinner


# Function to verify the URL
# This function checks if the URL is valid and reachable
# It takes a URL as input and prints an error message if the URL is invalid or unreachable
@spinner("Verifying URL")
def verifyUrl(url):
    """
    Verify if the given URL is valid and reachable.
    :param url: The URL to verify
    :return: website domain name
    """
    print(colored("🔹 Verifying URL...", "yellow"))
    # Check if the URL is empty
    if url.strip() == "":
        print(colored("Please provide a valid URL.", "red"))
        return
    # Check if the website URL exists
    if not url:
        print(colored("Please provide a valid URL.", "red"))
        return
    # Check if the URL is valid
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
        # print(colored("Please provide a valid URL starting with http:// or https://", "red"))
        # return
    # Check if the URL is reachable
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(colored(f"Website is not reachable. Status code: {response.status_code}", "red"))
            return
    except requests.exceptions.RequestException as e:
        print(colored(f"Error reaching the website: {e}", "red"))
        return
    print(colored("🔹 URL is valid and reachable.", "green"))

    # Extract the domain name from the URL
    domain_name = url.split("//")[-1].split("/")[0]
    print(colored(f"🔹 Domain name extracted: {domain_name}", "green"))
    return domain_name