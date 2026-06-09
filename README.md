## Flood Susceptbility Mapping with Logistic Regression

This project focuses on classifying flood-prone areas using Logistic Regression. The dataset focused on topographic depth, rainfall and distance to river features to predict flood risk

**Application Demo:**

https://github.com/user-attachments/assets/175de547-811b-4208-8d25-70a7592ca5dd



## Running the Code
1. Clone the repository
```sh
git clone https://github.com/Sherwin-77/my-banjir
cd my-banjir
```


2. Install the required packages

>[!NOTE]
> It is recommended to use venv here
>```sh
>python -m venv venv
>source venv/bin/activate  # On Windows use `venv\Scripts\activate`
>```

```sh
pip install -r requirements.txt
```

3. Run the init script to train model and download tiles

```sh
python init.py
```

4. Run the Streamlit app

```sh
streamlit run app.py
```
