# Proyecto-SegFormer-Analitica-de-Datos

## How to use locally

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
