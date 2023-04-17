# Interpretable-Gradient-Boosting
Interpretable Gradient Boosting - Real Estate House Price Prediction
Ask a home buyer to describe their dream house, and they probably won’t begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition’s dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence. With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

Installation instructions I followed: 
I installed Docker and WSL2 through the links provided on the repo-website, https://www.docker.com/products/docker-desktop and https://docs.microsoft.com/en-us/windows/wsl/install-win10. I already had VSC installed but needed to install the Docker and WSL extensions. I made a new repository on GitHub and cloned it to my desktop (along with adding my ssh-key to .git/config). I created app.py and Dockerfile to test my setup. It worked in a sufficient manner, but I am unsure where WSL2 comes into play in this setup.

![Docker Setup Proof](DockerSetupImage.png "Docker Setup Image")

I had a lot of troubles with HF and many of my classmates did as well. We came to the consensus that it would be easier and best to just use Streamlit Cloud as did the example that was sent to us. We hope this is sufficient. The model training is a little slow so sorry about that, but you might need to wait a bit before the full website loads.
LINK TO WORKING STREAMLIT SPACE FOR MILESTONE-3: https://fortisn7-interpretable-gradient-boosting-streamlitapp-mi-1sb1gu.streamlit.app