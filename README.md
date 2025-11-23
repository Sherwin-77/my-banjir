## Banjir Classification Based on Logistic Regression

This project focuses on classifying flood-prone areas using Logistic Regression. The dataset focused on topographic depth, rainfall and distance to river features to predict flood risk

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

3. Run the main script
There are 2 models available here: Linear Logistic Regression and Logistic Regression with Polynomial Features. You can run either of them by executing the respective script.
```sh
python main.py
# or
python main_polynomial.py
```