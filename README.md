<p align="center">
    <img src="https://readme-typing-svg.herokuapp.com?font=Time+New+Roman&color=cyan&size=25&center=true&vCenter=true&width=600&height=100&lines=SegFormer Implementation;Computer Vision;">
</p>

### What is SegFormer?

SegFormer is a cutting-edge semantic segmentation framework developed by NVIDIA, designed to unify Transformers with lightweight multilayer perceptron (MLP) decoders. It features a novel hierarchical Transformer encoder that outputs multiscale features without the need for positional encoding, enhancing adaptability to varying input resolutions. The MLP decoder aggregates information from different layers, combining both local and global attention mechanisms to produce powerful representations.

> For an in-depth understanding, refer to the [original paper](https://arxiv.org/abs/2105.15203)

### Segmentation

Below is an example of semantic segmentation using a pre-trained SegFormer model:

![Segmentation Example](data/images/001_segformer_img.png)


###  How to use locally

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/PaolaChaux/Proyecto-SegFormer-Analitica-de-Datos.git
   cd Proyecto-SegFormer-Analitica-de-Datos
   ```

> [!TIP]
> In the next step is highly recommended to use uv, since its way faster and the hole project was developed using uv. CHeck the docs [here](https://docs.astral.sh/uv/#uv)

2. Setup enviroment & requirements

    <details>
    <summary><b>Using <a href=https://docs.astral.sh/uv/>UV</a>></b></summary>

    - Install uv
        ```
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```
    - Create a uv project
        ```
        uv init
        ```
    - Install dependencies:
        ```bash
        uv pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
        ```
    - Run the app
        ```
        uv run streamlit run src/appStreamlit.py
        ```
    </details>

    <details>
    <summary><b>Using pip</b></summary>

    - Create the enviroment
        ```
        python -m venv .venv
        ```
    - Activate the enviroment
        ```
        source .venv/bin/activate # Linux
        ```
        ```
        .venv\Scripts\activate # Windows
        ```
    - Install dependencies
        ```
        pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
        ```
    - Run the app
        ```
        streamlit run src/Homepage.py
        ```
    </details>

> [!NOTE]
> Keep in mind that the Dockerfile used to deploy its based on an image with uv preinstall.

### Additional Docs

- If you want to know more about the SegFormer and its architecture you can either run the streamlit app or refer to [here](docs/Architecture.md)

- If you want to know how was the deployment process in GCP, please refer to [here](docs/Deploy.md)
