import streamlit as st
import pandas as pd
import sqlite3
from xgboost import XGBRegressor
import datetime
import pickle
import joblib
import numpy as np
from contextlib import contextmanager
from sklearn.preprocessing import LabelEncoder, StandardScaler
from matplotlib import pyplot as plt

@contextmanager
def get_db_connection(database='BigBasket.db'):
    conn = sqlite3.connect(database)
    try:
        yield conn
    finally:
        conn.close()

def read_table(table_name):
    with get_db_connection() as conn:
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)

@st.cache_resource
def load_model(model_path, demand_path, encoder_path):
    model = XGBRegressor()
    model.load_model(model_path)
    demand_model = joblib.load(open(demand_path, 'rb'))
    with open(encoder_path, 'rb') as f:
        le_product = pickle.load(f)
    return model, demand_model, le_product

def load_initial_data():
    managers_df = read_table("managers")
    customers_df = read_table("customers")
    products_df = read_table("ProductsOnWebsite")
    orders_df = read_table("Orders")
    return managers_df, customers_df, products_df, orders_df

session_state_keys = [
    'cart', 'recommended_cart', 'view_mode', 'logged_in', 'user_type', 
    'page', 'username', 'show_add_product_form', 'show_edit_form',
    'search_query_manager', 'current_page', 'manager_current_page', 
    'selected_product_edit_key', 'show_edit_product_form', 
    'search_query_customer'
]
for key in session_state_keys:
    if key not in st.session_state:
        st.session_state[key] = None

st.session_state['cart'] = st.session_state.get('cart', [])
st.session_state['recommended_cart'] = st.session_state.get('recommended_cart', [])
st.session_state['view_mode'] = st.session_state.get('view_mode', 'products')
st.session_state['logged_in'] = st.session_state.get('logged_in', False)
st.session_state['user_type'] = st.session_state.get('user_type', None)
st.session_state['page'] = st.session_state.get('page', 'Login')
st.session_state['show_add_product_form'] = st.session_state.get('show_add_product_form', False)
st.session_state['show_edit_form'] = st.session_state.get('show_edit_form', False)
st.session_state['current_page'] = st.session_state.get('current_page', 1) or 1
st.session_state['manager_current_page'] = st.session_state.get('manager_current_page', 1) or 1

def truncate_text(text, max_length):
    return text[:max_length] + "..." if len(text) > max_length else text

def inject_custom_css():
    st.markdown(
        """
        <style>
            [data-testid="stAppViewContainer"] {
                background-color: #595959;
            }
            [data-testid="stSidebar"] [data-testid="stImage"] {
                margin-top: 0px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_customer_header():
    col1, col2, col3, col4, col5, col6 = st.columns([1, 3, 7, 3, 3, 3])
    with col1:
        st.image("bb.jpeg", width=80)
    with col2:
        st.write("")
        st.write("Fresh Market")
        st.write("Online Grocery Store")
    with col3:
        search_query = st.text_input("", value="", placeholder="Search for products", key="search")
    with col4:
        st.write("")
        st.write("")
        if st.button('Search'):
            st.session_state['search_query_customer'] = search_query
            st.session_state['view_mode'] = 'search'
            st.rerun()
    with col5:
        st.write("")
        st.write("")
        if st.button('Profile'):
            st.session_state['view_mode'] = 'profile'
            st.rerun()
    with col6:
        st.write("")
        st.write("")
        if st.button('Logout', key='customer_logout'):
            st.session_state['logged_in'] = False
            st.session_state['user_type'] = None
            st.session_state['page'] = "Login"
            st.rerun()
    menu_col1, menu_col2, menu_col3, menu_col4, menu_col5, menu_col6 = st.columns([10, 5, 3, 3, 3, 1])
    with menu_col1:
        categories = ["All Products"] + sorted(st.session_state['products_df']['Category'].unique().tolist())
        selected_category = st.selectbox("Category", categories, key="category_select_customer")
        st.session_state['filtered_products_df'] = st.session_state['products_df'] if selected_category == "All Products" else st.session_state['products_df'][st.session_state['products_df']['Category'] == selected_category]

    if 'search_query_customer' in st.session_state and st.session_state['search_query_customer']:
        st.session_state['filtered_products_df'] = st.session_state['filtered_products_df'][st.session_state['filtered_products_df']['ProductName'].str.contains(st.session_state['search_query_customer'], case=False)]
    with menu_col2:
        st.write("")
        st.write("")
        if st.button("Recommended Cart"):
            st.session_state['view_mode'] = 'recommended_cart'
            st.rerun()
    with menu_col3:
        st.write("")
        st.write("")
        if st.button('View Cart'):
            st.session_state['view_mode'] = 'cart'
            st.rerun()
    with menu_col4:
        st.write("")
        st.write("")
        if st.button('Shopping History'):
            st.session_state['view_mode'] = 'shopping_history'
            st.rerun()
    with menu_col5:
        st.write("")
        st.write("")
        if st.button('HomePage'):
            st.session_state['view_mode'] = 'products'
            st.session_state['search_query_customer'] = None
            st.rerun()
    with menu_col6:
        pass

def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    user_type = st.selectbox("User Type", ["Customer", "Manager"])
    if st.button("Login"):
        if username and password:
            user_df = st.session_state['customers_df'] if user_type == "Customer" else st.session_state['managers_df']
            if username in user_df['name'].values:
                user_data = user_df[user_df['name'] == username]
                if password == user_data['password'].values[0]:
                    st.session_state['logged_in'] = True
                    st.session_state['user_type'] = user_type
                    st.session_state['username'] = username
                    st.session_state['view_mode'] = 'manager_products' if user_type == "Manager" else 'products'
                    st.rerun()
                else:
                    st.error("Incorrect password")
            else:
                st.error("Username does not exist")
        else:
            st.error("Please fill in all fields")

    if st.button("Register New User"):
        st.session_state['page'] = "Registration"
        st.rerun()

def registration_page(customers_df):
    st.title("Registration")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if st.button("Register"):
        if new_username and new_password:
            if new_password == confirm_password:
                if new_username in customers_df['name'].values:
                    st.error("Username already exists")
                else:
                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO customers (name, password) VALUES (?, ?)", (new_username, new_password))
                        conn.commit()
                    st.cache_data.clear()
                    st.session_state['customers_df'] = read_table("customers")
                    st.success("Registration successful!")
                    st.session_state['logged_in'] = False
                    st.session_state['user_type'] = "Customer"
                    st.session_state['page'] = "Login"
                    st.rerun()
            else:
                st.error("Passwords do not match")
        else:
            st.error("Please fill in all fields")

    if st.button("Already have an account? Login"):
        st.session_state['page'] = "Login"
        st.rerun()

def view_cart():
    st.title("Shopping Cart")
    col1, col2, col3, col4, col5, col6 = st.columns([1, 3, 7, 3, 3, 3])
    with col1:
        st.image("bb.jpeg", width=80)
    with col2:
        st.write("")
        st.write("Fresh Market")
        st.write("Online Grocery Store")
    with col6:
        if st.button("Back to Homepage"):
            st.session_state['view_mode'] = 'products'
            st.rerun()
    if not st.session_state['cart']:
        st.warning("Your cart is empty.")
    else:
        cart_dict = {}
        for product_name in st.session_state['cart']:
            product_name = product_name.get('ProductName', '') if isinstance(product_name, dict) else product_name
            cart_dict[product_name] = cart_dict.get(product_name, 0) + 1
        for product_name, quantity in cart_dict.items():
            product_details = st.session_state['products_df'][st.session_state['products_df']['ProductName'] == product_name].iloc[0]
            cols = st.columns([3, 1, 1, 1, 2])
            with cols[0]:
                st.write(product_name)
                if 'Image_Url' in product_details:
                    st.image(product_details['Image_Url'], width=150)
            with cols[1]:
                if st.button("\-", key=f'decrease_{product_name}'):
                    if quantity > 1:
                        st.session_state['cart'].remove(product_name)
                    else:
                        st.session_state['cart'] = [item for item in st.session_state['cart'] if item != product_name]
                    st.rerun()
            with cols[2]:
                st.write(f"Quantity: {quantity}")
            with cols[3]:
                if st.button("\+", key=f'increase_{product_name}'):
                    st.session_state['cart'].append(product_name)
                    st.rerun()
            with cols[4]:
                if st.button("Remove", key=f'remove_{product_name}'):
                    st.session_state['cart'] = [item for item in st.session_state['cart'] if item != product_name]
                    st.rerun()
        if st.button("Proceed to Checkout"):
            st.session_state['view_mode'] = 'checkout'
            st.rerun()

def checkout():
    st.title("Checkout")
    if st.button("Back to Homepage"):
        st.session_state['view_mode'] = 'products'
        st.rerun()
    if not st.session_state['cart']:
        st.warning("Your cart is empty.")
        if st.button("Back to Products"):
            st.session_state['view_mode'] = 'products'
            st.rerun()
        return
    checkout_columns = ["Product", "Quantity", "Price per Unit", "Product Total"]
    cart_items_data = []
    total_cost = 0
    processed_cart = [item['ProductName'] if isinstance(item, dict) and 'ProductName' in item else item for item in st.session_state['cart']]
    for product_name in set(processed_cart):
        product_details = st.session_state['products_df'][st.session_state['products_df']['ProductName'] == product_name].iloc[0]
        quantity = processed_cart.count(product_name)
        price_per_unit = product_details['Price']
        product_total = quantity * price_per_unit
        total_cost += product_total
        cart_items_data.append([product_name, quantity, price_per_unit, product_total])
    cart_items_df = pd.DataFrame(cart_items_data, columns=checkout_columns)
    st.table(cart_items_df)
    st.markdown(f"**Total Cost: ₹{total_cost:.2f}**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Confirm Order"):
            today_date = datetime.datetime.now().strftime('%d/%m/%Y')
            order_id = generate_order_id(st.session_state['username'])
            customer_id = st.session_state['username']
            with get_db_connection() as conn:
                cursor = conn.cursor()
                for item in cart_items_data:
                    cursor.execute(
                        "INSERT INTO Orders (CustomerID, OrderID, ProductName, Quantity, OrderDate, Price) VALUES (?, ?, ?, ?, ?, ?)",
                        (customer_id, order_id, item[0], item[1], today_date, item[2])
                    )
                conn.commit()
            st.cache_data.clear()
            st.session_state['orders_df'] = read_table("Orders")
            st.success("Thank you for your order! Your purchase has been added to your shopping history.")
            st.session_state['cart'] = []
            st.session_state['view_mode'] = 'products'
            st.rerun()
    with col2:
        if st.button("Back to Cart"):
            st.session_state['view_mode'] = 'cart'
            st.rerun()

def generate_order_id(username):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Orders WHERE CustomerID=?", (username,))
        order_count = cursor.fetchone()[0]
    return f"{username}-{order_count + 1}"

def display_products(products_df):
    products_per_page = 42
    total_products = len(products_df)
    total_pages = (total_products + products_per_page - 1) // products_per_page
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 1
    start_idx = (st.session_state['current_page'] - 1) * products_per_page
    end_idx = start_idx + products_per_page
    products_to_display = products_df.iloc[start_idx:end_idx]
    num_columns = 6
    for i in range(0, len(products_to_display), num_columns):
        cols = st.columns(num_columns)
        for j, col in enumerate(cols):
            product_idx = i + j
            if product_idx < len(products_to_display):
                product = products_to_display.iloc[product_idx]
                with col:
                    truncated_product_name = truncate_text(product['ProductName'], 25)
                    with st.expander(truncated_product_name):
                        st.write(product['ProductName'])
                    st.image(product['Image_Url'], use_column_width=True)
                    st.write(f"Price: ₹{product['Price']}")
                    st.write(f"After Discount: ₹{product['DiscountPrice']}")
                    if st.button("Add to Cart", key=f'Add_to_Cart_{product_idx}'):
                        if st.session_state['cart'] is None:
                            st.session_state['cart'] = []
                        st.session_state['cart'].append(product['ProductName'])
                        st.success(f"Added {truncate_text(product['ProductName'], 13)} to cart")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Previous") and st.session_state['current_page'] > 1:
            st.session_state['current_page'] -= 1
            st.rerun()
    with col2:
        st.write(f"Page {st.session_state['current_page']} of {total_pages}")
    with col3:
        if st.button("Next") and st.session_state['current_page'] < total_pages:
            st.session_state['current_page'] += 1
            st.rerun()

def display_manager_products(filtered_products_df):
    products_per_page = 42
    total_products = len(filtered_products_df)
    total_pages = (total_products + products_per_page - 1) // products_per_page
    if 'manager_current_page' not in st.session_state:
        st.session_state['manager_current_page'] = 1
    start_idx = (st.session_state['manager_current_page'] - 1) * products_per_page
    end_idx = start_idx + products_per_page
    products_to_display = filtered_products_df.iloc[start_idx:end_idx]
    num_columns = 6
    for i in range(0, len(products_to_display), num_columns):
        cols = st.columns(num_columns)
        for j, col in enumerate(cols):
            product_idx = i + j
            if product_idx < len(products_to_display):
                product = products_to_display.iloc[product_idx]
                with col:
                    truncated_product_name = truncate_text(product['ProductName'], 23)
                    with st.expander(truncated_product_name):
                        st.write(f"Name: {product['ProductName']}")
                        st.write(f"Quantity: {product['Quantity']}")
                    st.image(product['Image_Url'], use_column_width=True)
                    st.write(f"Price: ₹{product['Price']}")
                    st.write(f"After Discount: ₹{product['DiscountPrice']}")

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        unique_edit_key = f'edit_{product["ProductName"]}_{product["Quantity"]}'
                        if st.button("Edit", key=unique_edit_key):
                            st.session_state['selected_product_edit_key'] = f'{product["ProductName"]}_{product["Quantity"]}'
                            st.session_state['show_edit_form'] = True
                            st.rerun()
                    with col2:
                        unique_delete_key = f'delete_{product["ProductName"]}_{product["Quantity"]}'
                        if st.button("Delete", key=unique_delete_key):
                            delete_product(product["ProductName"], product["Quantity"])
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Previous") and st.session_state['manager_current_page'] > 1:
            st.session_state['manager_current_page'] -= 1
            st.rerun()
    with col2:
        st.write(f"Page {st.session_state['manager_current_page']} of {total_pages}")
    with col3:
        if st.button("Next") and st.session_state['manager_current_page'] < total_pages:
            st.session_state['manager_current_page'] += 1
            st.rerun()

def delete_product(product_name, quantity):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM ProductsOnWebsite WHERE ProductName = ? AND Quantity = ?", (product_name, quantity))
        conn.commit()
        st.sidebar.success("Product deleted successfully!")
        st.cache_data.clear()
        st.session_state['products_df'] = read_table("ProductsOnWebsite")
        st.rerun()

def manager_welcome_page(products_df):
    if st.session_state.get('view_mode', '') in ['products', 'manager_products']:
        render_manager_header()
        menu_col1, menu_col2, menu_col3, menu_col4, menu_col5 = st.columns([5, 5, 3, 3, 3])
        with menu_col1:
            categories = ["All Products"] + sorted(products_df['Category'].unique().tolist())
            selected_category = st.selectbox("Category", categories, key="category_select_manager")
            if selected_category != "All Products":
                products_df = products_df[products_df['Category'] == selected_category]
            filtered_products_df = products_df

    with st.sidebar:
        st.header("Management Options")
        if st.button("Add New Product"):
            st.session_state['show_add_product_form'] = True
            st.session_state['show_edit_product_form'] = False
        elif 'show_add_product_form' not in st.session_state:
            st.session_state['show_add_product_form'] = False
        if 'show_edit_product_form' not in st.session_state:
            st.session_state['show_edit_product_form'] = False
        if st.button('Admin Registration'):
            st.session_state['view_mode'] = 'admin_registration'
            st.rerun()
        if st.button('Manage Customers'):
            st.session_state['view_mode'] = 'manage_customer'
            st.rerun()
        if st.button('Demand Forecasting Panel'):
            st.session_state['view_mode'] = 'demand_forecasting'
            st.rerun()

    if st.session_state.get('view_mode', '') == 'manager_products':
        display_manager_products(st.session_state['filtered_products_df'])
    elif st.session_state.get('view_mode', '') == 'admin_registration':
        admin_registration()
    elif st.session_state.get('view_mode', '') == 'manage_customer':
        manage_customer()
    elif st.session_state.get('view_mode', '') == 'demand_forecasting':
        demand_forecasting()
    if st.session_state['show_add_product_form']:
        add_new_product()
    if st.session_state['show_edit_form']:
        edit_product()

def edit_product():
    if 'selected_product_edit_key' in st.session_state:
        selected_product_key = st.session_state['selected_product_edit_key']
        product_name, quantity = selected_product_key.split('_', 1)
        product_details = st.session_state['products_df'].loc[(st.session_state['products_df']['ProductName'] == product_name) & (st.session_state['products_df']['Quantity'] == quantity)].iloc[0]
        if st.session_state.get('show_edit_form', False):
            st.sidebar.subheader("Edit Product Details")
            with st.sidebar.form(key=f'edit_product_form_{selected_product_key}'):
                new_product_name = st.text_input("Product Name", value=product_details['ProductName'], key=f"edit_new_product_name_{selected_product_key}")
                new_price = st.number_input("Price", value=float(product_details['Price']), min_value=0.0, key=f"edit_new_price_{selected_product_key}")
                new_discount_price = st.number_input("Discount Price", value=float(product_details['DiscountPrice']), min_value=0.0, key=f"edit_new_discount_price_{selected_product_key}")
                new_image_url = st.text_input("Image URL", value=product_details['Image_Url'], key=f"edit_new_image_url_{selected_product_key}")
                all_fields_filled = new_product_name and new_image_url
                not_duplicate = not (new_product_name != product_details['ProductName'] and new_product_name in st.session_state['products_df']['ProductName'].values)
                valid_url = new_image_url.startswith("http://") or new_image_url.startswith("https://")
                submit_button = st.form_submit_button("Save Changes")
            if submit_button and all_fields_filled and not_duplicate and valid_url:
                if st.sidebar.button("Confirm Update"):
                    update_product(product_name, quantity, new_product_name, new_price, new_discount_price, new_image_url)
                    st.sidebar.success("Product updated successfully!")
                    st.session_state['show_edit_form'] = False
                    del st.session_state['selected_product_edit_key']
                    st.rerun()
                else:
                    st.sidebar.error("Please confirm the changes.")
            else:
                if not all_fields_filled:
                    st.sidebar.error("All fields must be filled.")
                if not not_duplicate:
                    st.sidebar.error("Product name already exists. Please use a different name.")
                if not valid_url:
                    st.sidebar.error("Please enter a valid URL.")
            if st.sidebar.button("Cancel Edit"):
                st.session_state['show_edit_form'] = False
                del st.session_state['selected_product_edit_key']
                st.rerun()
        else:
            st.session_state['show_edit_form'] = True
            st.session_state['show_add_product_form'] = False
            st.rerun()

def update_product(old_name, old_quantity, new_name, new_price, new_discount_price, new_image_url):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE ProductsOnWebsite
            SET ProductName = ?, Price = ?, DiscountPrice = ?, Image_Url = ?
            WHERE ProductName = ? AND Quantity = ?""", (new_name, new_price, new_discount_price, new_image_url, old_name, old_quantity))
        conn.commit()
    st.cache_data.clear()
    st.session_state['products_df'] = read_table("ProductsOnWebsite")
    del st.session_state['selected_product_edit_key']
    st.rerun()

def welcome_page():
    if st.session_state['user_type'] == "Customer":
        if st.session_state['view_mode'] in ['products', 'search']:
            render_customer_header()
            display_products(st.session_state['filtered_products_df'])
        if st.session_state['view_mode'] == 'cart':
            view_cart()
        elif st.session_state['view_mode'] == 'shopping_history':
            view_shopping_history(st.session_state['username'])
        elif st.session_state['view_mode'] == 'recommended_cart':
            view_recommended_cart()
        elif st.session_state['view_mode'] == 'profile':
            update_profile()


def add_new_product():
    if st.session_state.get('show_add_product_form', False):
        st.sidebar.subheader("Add New Product")
        with st.sidebar.form(key='add_product_form'):
            new_product_name = st.text_input("Product Name", key="new_product_name")
            new_quantity = st.text_input("Brand", key="new_quantity")
            new_price = st.number_input("Price", min_value=0.0, key="new_price")
            new_discount_price = st.number_input("Discount Price", min_value=0.0, key="new_discount_price")
            new_category = st.text_input("Category", key="new_category")
            new_sub_category = st.text_input("SubCategory", key="new_sub_category")
            new_image_url = st.text_input("Image URL", key="new_image_url")
            new_absolute_url = st.text_input("Absolute URL", key="new_absolute_url")
            submit_button = st.form_submit_button("Add Product")
        if submit_button:
            all_fields_filled = all([new_product_name, new_quantity, new_category, new_sub_category, new_image_url, new_absolute_url])
            is_quantity_valid = new_quantity.isdigit() and int(new_quantity) > 0
            is_urls_valid = all([url.startswith("http://") or url.startswith("https://") for url in [new_image_url, new_absolute_url]])
            not_duplicate_name = new_product_name not in st.session_state['products_df']['ProductName'].values
            if all_fields_filled and is_quantity_valid and is_urls_valid and not_duplicate_name:
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO ProductsOnWebsite (ProductName, Quantity, Price, DiscountPrice, Category, SubCategory, Image_Url, Absolute_Url)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", 
                        (new_product_name, new_quantity, new_price, new_discount_price, new_category, new_sub_category, new_image_url, new_absolute_url))
                    conn.commit()
                    st.sidebar.success("Product added successfully!")
                    st.cache_data.clear()
                    st.session_state['products_df'] = read_table("ProductsOnWebsite")
                    del st.session_state['show_add_product_form']
                    st.rerun()
            else:
                if not all_fields_filled:
                    st.sidebar.error("All fields must be filled.")
                if not is_quantity_valid:
                    st.sidebar.error("Quantity must be a positive integer.")
                if not is_urls_valid:
                    st.sidebar.error("Please enter valid URLs starting with http:// or https://")
                if not not_duplicate_name:
                    st.sidebar.error("Product name already exists. Please use a different name.")
        if st.sidebar.button("Cancel Add"):
            del st.session_state['show_add_product_form']
            st.rerun()

def view_shopping_history(username):
    st.title(f"Shopping History for {username}")
    col1, col2, col3, col4, col5, col6 = st.columns([1, 3, 7, 3, 3, 3])
    with col1:
        st.image("bb.jpeg", width=80)
    with col2:
        st.write("")
        st.write("Fresh Market")
        st.write("Online Grocery Store")
    with col6:
        if st.button("Back to Homepage"):
            st.session_state['view_mode'] = 'products'
            st.rerun()
    try:
        st.session_state['orders_df']['OrderDate'] = pd.to_datetime(st.session_state['orders_df']['OrderDate'], format='%d/%m/%Y').dt.date
    except ValueError as e:
        st.error(f"Date format error: {str(e)}")
        return
    user_orders = st.session_state['orders_df'][st.session_state['orders_df']['CustomerID'] == username].copy()
    user_orders.sort_values(by='OrderDate', ascending=False, inplace=True)
    if user_orders.empty:
        st.warning("You have no shopping history.")
    else:
        grouped_orders = user_orders.groupby('OrderID', sort=False)
        for order_id, order_details in grouped_orders:
            order_date = order_details['OrderDate'].iloc[0].strftime('%d/%m/%Y')
            st.markdown(f"### Order ID: {order_id} - Date: {order_date}")
            order_details['Price'] = order_details['Price'].apply(lambda x: f"₹{x:.2f}")
            display_cols = ['ProductName', 'Quantity', 'Price']
            st.table(order_details[display_cols].reset_index(drop=True))
            total_cost = order_details['Price'].replace('₹', '', regex=True).astype(float).sum()
            st.write(f"Total Cost: ₹{total_cost:.2f}")
            st.write("---")

def view_recommended_cart():
    st.title("Top 10 Recommended Products in Your Cart")
    col1, col2, col3, col4, col5, col6 = st.columns([1, 3, 7, 3, 3, 3])
    with col1:
        st.image("bb.jpeg", width=80)
    with col2:
        st.write("")
        st.write("Fresh Market")
        st.write("Online Grocery Store")
    with col6:
        if st.button("Back to Homepage"):
            st.session_state['view_mode'] = 'products'
            st.rerun()
    customer_id = st.session_state['username']
    if st.session_state['recommended_cart'] is None:
        st.session_state['recommended_cart'] = []

    st.session_state['orders_df']['ProductID'] = le_product.transform(st.session_state['orders_df']['ProductName'])

    customer_features = st.session_state['orders_df'].groupby('CustomerID').agg(
        TotalOrders=pd.NamedAgg(column='OrderID', aggfunc='nunique'),
        AvgQuantity=pd.NamedAgg(column='Quantity', aggfunc='mean'),
        MostBoughtProduct=pd.NamedAgg(column='ProductID', aggfunc=lambda x: x.mode()[0])
    ).reset_index()

    if customer_id in customer_features['CustomerID'].values:
        customer_data = customer_features[customer_features['CustomerID'] == customer_id].iloc[0]
        product_names = st.session_state['products_df']['ProductName'].unique()
        product_ids, unique_indices = np.unique(le_product.transform(product_names), return_index=True)
        product_images = st.session_state['products_df']['Image_Url'].values[unique_indices]
        sample_customer_data = pd.DataFrame({
            'ProductID': product_ids,
            'Month': [pd.Timestamp('today').month] * len(product_ids),
            'DayOfWeek': [pd.Timestamp('today').dayofweek] * len(product_ids),
            'TotalOrders': [customer_data['TotalOrders']] * len(product_ids),
            'AvgQuantity': [customer_data['AvgQuantity']] * len(product_ids),
            'MostBoughtProduct': [customer_data['MostBoughtProduct']] * len(product_ids)
        })
        predicted_quantities = model.predict(sample_customer_data)
        sample_customer_data['ProductName'] = product_names
        sample_customer_data['ProductImage'] = product_images
        sample_customer_data['PredictedQuantity'] = predicted_quantities
        top_recommendations = sample_customer_data.sort_values(by='PredictedQuantity', ascending=False).head(10)
        
        if top_recommendations.empty:
            st.warning("You need to make orders first to get a recommended cart! 😊")
        else:
            for product in top_recommendations['ProductName']:
                if product not in st.session_state['recommended_cart']:
                    st.session_state['recommended_cart'].append(product)
                    
            updated_recommended_cart = list(st.session_state['recommended_cart'])
            for _, row in top_recommendations.iterrows():
                cols = st.columns([3, 1, 1, 1, 1, 2])
                with cols[0]:
                    st.write(row['ProductName'])
                    recommended_product_index = np.where(st.session_state['products_df']['ProductName'] == row['ProductName'])[0][0]
                    st.image(st.session_state['products_df']['Image_Url'].iloc[recommended_product_index], width=150)
                with cols[1]:
                    if st.button("\-", key=f'decrease_{row["ProductName"]}'):
                        if updated_recommended_cart.count(row['ProductName']) > 1:
                            updated_recommended_cart.remove(row['ProductName'])
                        else:
                            updated_recommended_cart = [item for item in updated_recommended_cart if item != row['ProductName']]
                        st.session_state['recommended_cart'] = updated_recommended_cart
                        st.rerun()
                with cols[2]:
                    st.write(f"Quantity: {updated_recommended_cart.count(row['ProductName'])}")
                with cols[3]:
                    if st.button("\+", key=f'increase_{row["ProductName"]}'):
                        updated_recommended_cart.append(row['ProductName'])
                        st.session_state['recommended_cart'] = updated_recommended_cart
                        st.rerun()
                with cols[4]:
                    if st.button("Remove", key=f'remove_{row["ProductName"]}'):
                        updated_recommended_cart = [item for item in updated_recommended_cart if item != row['ProductName']]
                        st.session_state['recommended_cart'] = updated_recommended_cart
                        st.rerun()

            if st.button("Add to Main Cart"):
                if st.session_state['cart'] is None:
                    st.session_state['cart'] = []
                st.session_state['cart'].extend(st.session_state['recommended_cart'])
                st.session_state['recommended_cart'] = []
                st.session_state['add_to_cart_success'] = True
                st.rerun()
    else:
        st.warning("You need to make orders first to get a recommended cart! 😊")

    if st.session_state.get('add_to_cart_success', False):
        st.success("Recommended products added to your cart.")
        st.session_state['add_to_cart_success'] = False

def render_manager_header():
    col1, col2, col3, col4, col5 = st.columns([1, 3, 7, 3, 3])
    with col1:
        st.image("bb.jpeg", width=80)
    with col2:
        st.write("")
        st.write("Fresh Market")
        st.write("Online Grocery Store")
    with col3:
        search_query = st.text_input("", value="", placeholder="Search for products", key="manager_search")
    with col4:
        st.write("")
        st.write("")
        if st.button('Search'):
            st.session_state['search_query_manager'] = search_query.strip()
            st.rerun()
    with col5:
        st.write("")
        st.write("")
        if st.button('Logout', key='manager_logout'):
            st.session_state['logged_in'] = False
            st.session_state['user_type'] = None
            st.session_state['page'] = "Login"
            st.rerun()

    if 'search_query_manager' in st.session_state and st.session_state['search_query_manager']:
        st.session_state['filtered_products_df'] = st.session_state['products_df'][st.session_state['products_df']['ProductName'].str.contains(st.session_state['search_query_manager'], case=False)]
    else:
        st.session_state['filtered_products_df'] = st.session_state['products_df']

def update_profile():
    st.title("Update Profile")
    col1, col2, col3, col4, col5, col6 = st.columns([1, 3, 7, 3, 3, 3])
    with col1:
        st.image("bb.jpeg", width=80)
    with col2:
        st.write("")
        st.write("Fresh Market")
        st.write("Online Grocery Store")
    with col6:
        if st.button("Back to Homepage"):
            st.session_state['view_mode'] = 'products'
            st.rerun()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customers WHERE name = ?", (st.session_state['username'],))
        current_user = cursor.fetchone()
    if current_user:
        with st.form("update_form"):
            new_username = st.text_input("New Username", value=current_user[0])
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            update_button = st.form_submit_button("Update")
        if update_button:
            if not new_username or not new_password or not confirm_password:
                st.error("Please fill in all fields.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM customers WHERE name = ?", (new_username,))
                    if cursor.fetchone() and new_username != current_user[0]:
                        st.error("Username already exists. Please choose another username.")
                    else:
                        try:
                            cursor.execute("UPDATE customers SET name = ?, password = ? WHERE name = ?", (new_username, new_password, current_user[0]))
                            conn.commit()
                            st.cache_data.clear()
                            st.success("Profile updated successfully!")
                            st.session_state['username'] = new_username
                            st.session_state['customers_df'] = read_table("customers")
                        except sqlite3.Error as e:
                            st.error(f"An error occurred: {e}")

def admin_registration():
    st.title("Manager Registration")
    if st.button('HomePage', key='admin_registration_homepage'):
        st.session_state['view_mode'] = 'manager_products'
        st.rerun()
    new_username = st.text_input("New Manager Username")
    new_password = st.text_input("New Manager Password", type="password")
    confirm_password = st.text_input("Confirm Manager Password", type="password")
    if st.button("Register Manager"):
        if new_username and new_password:
            if new_password == confirm_password:
                if new_username in st.session_state['managers_df']['name'].values:
                    st.error("Manager username already exists")
                else:
                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO managers (name, password) VALUES (?, ?)", (new_username, new_password))
                        conn.commit()
                        st.cache_data.clear()
                        st.success("Manager registration successful!")
                        st.session_state['logged_in'] = True
                        st.session_state['user_type'] = "Manager"
                        st.session_state['username'] = new_username
                        st.session_state['managers_df'] = read_table("managers")
                        st.session_state['view_mode'] = 'manager_products'
                        st.rerun()
            else:
                st.error("Passwords do not match")
        else:
            st.error("Please fill in all fields")

def manage_customer():
    st.title("Manage Customer")
    if st.button('HomePage', key='manage_customer_homepage'):
        st.session_state['view_mode'] = 'manager_products'
        st.rerun()
    customers = st.session_state['customers_df']
    selected_customer = st.selectbox("Select a customer", customers['name'])
    customer_details = customers[customers['name'] == selected_customer].iloc[0]
    st.write(f"Current Name: {customer_details['name']}")
    new_name = st.text_input("New Name", value=customer_details['name'])
    new_password = st.text_input("New Password", type="password")
    if st.button("Update"):
        if new_name and new_password:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE customers SET name = ?, password = ? WHERE name = ?", (new_name, new_password, selected_customer))
                conn.commit()
                st.cache_data.clear()
                st.success("Customer details updated successfully!")
                st.session_state['customers_df'] = read_table("customers")
                st.rerun()
        else:
            st.error("Please fill in all fields.")
    if st.button("Delete"):
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM customers WHERE name = ?", (selected_customer,))
            conn.commit()
            st.cache_data.clear()
            st.success("Customer deleted successfully!")
            st.session_state['customers_df'] = read_table("customers")
            st.rerun()

def demand_forecasting():
    st.title("Demand Forecasting")
    if st.button("Back to Homepage", key='demand_forecasting_homepage'):
        st.session_state['view_mode'] = 'manager_products'
        st.rerun()
    st.write("""
    The Demand Forecasting panel helps you predict the future demand for various products 
    based on historical sales data. This can assist in inventory management and planning 
    for upcoming demand trends.
    """)

    with st.spinner("Loading data and calculating demand forecasts..."):
        st.session_state['orders_df']['OrderDate'] = pd.to_datetime(st.session_state['orders_df']['OrderDate'], format='%d/%m/%Y')
        daily_demand_df = st.session_state['orders_df'].groupby(['ProductName', 'OrderDate']).agg({
            'Quantity': 'sum',
            'Price': 'mean'
        }).reset_index()
        merged_df = pd.merge(daily_demand_df, st.session_state['products_df'], on='ProductName', how='left', suffixes=('_order', '_product'))
        merged_df['DiscountPrice'] = merged_df['DiscountPrice'].fillna(0)
        merged_df['Price_order'] = merged_df['Price_order'].fillna(0)
        merged_df['Price_product'] = merged_df['Price_product'].fillna(0)
        merged_df['PriceDiff'] = merged_df['Price_order'] - merged_df['DiscountPrice']
        if 'Quantity' not in merged_df.columns:
            merged_df['Quantity'] = merged_df['Quantity_order'].fillna(0)
        merged_df['OrderDay'] = merged_df['OrderDate'].dt.day
        merged_df['OrderMonth'] = merged_df['OrderDate'].dt.month
        merged_df['Price_Discount_Interaction'] = merged_df['Price_order'] * merged_df['DiscountPrice']
        merged_df['Lag_Quantity_1'] = merged_df.groupby('ProductName')['Quantity'].shift(1).fillna(0)
        merged_df['Lag_Quantity_2'] = merged_df.groupby('ProductName')['Quantity'].shift(2).fillna(0)
        merged_df['Lag_Quantity_3'] = merged_df.groupby('ProductName')['Quantity'].shift(3).fillna(0)
        merged_df['Rolling_Mean_3'] = merged_df.groupby('ProductName')['Quantity'].transform(lambda x: x.shift(1).rolling(window=3).mean()).fillna(0)
        label_encoders = {}
        for column in ['ProductName', 'Brand', 'Category', 'SubCategory']:
            le = LabelEncoder()
            merged_df[column] = le.fit_transform(merged_df[column])
            label_encoders[column] = le
        X = merged_df[['ProductName', 'Brand', 'Price_order', 'DiscountPrice', 'Category', 'SubCategory', 
                    'OrderDay', 'OrderMonth', 'PriceDiff', 'Price_Discount_Interaction', 
                    'Lag_Quantity_1', 'Lag_Quantity_2', 'Lag_Quantity_3', 'Rolling_Mean_3']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        predictions = demand_model.predict(X_scaled)
        merged_df['Predict Demand Score'] = predictions
        merged_df['ProductName'] = label_encoders['ProductName'].inverse_transform(merged_df['ProductName'])

    st.write("### Predicted Demand Overview")
    summary_df = merged_df.groupby('ProductName').agg({
        'Predict Demand Score': 'mean'
    }).reset_index()
    summary_df = summary_df.sort_values(by='Predict Demand Score', ascending=False).head(10)
    st.write("#### Top 10 Products by Predict Demand Score")
    st.write(summary_df[['ProductName', 'Predict Demand Score']])
    st.write("### Demand Forecast Visualization")
    product_selection = st.selectbox("Select a product to visualize", summary_df['ProductName'].unique())
    if product_selection:
        product_data = merged_df[(merged_df['ProductName'] == product_selection) & (merged_df['OrderDate'] >= pd.Timestamp('2024-01-01'))]
        st.write(f"#### Demand Prediction for {product_selection}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(product_data['OrderDate'], product_data['Predict Demand Score'], label='Predict Demand Score', color='blue')
        ax.set_xlabel('Date')
        ax.set_ylabel('Predict Demand Score')
        ax.set_title(f'Demand Prediction for {product_selection}')
        ax.legend()
        st.pyplot(fig)
        st.write(f"##### Detailed Prediction Data for {product_selection}")
        st.write(product_data[['OrderDate', 'Predict Demand Score']])


if __name__ == "__main__":
    st.set_page_config(page_title="Fresh Market", layout="wide", page_icon="bb.jpeg")
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['user_type'] = None
    if 'page' not in st.session_state:
        st.session_state['page'] = "Login"
    inject_custom_css()
    if 'managers_df' not in st.session_state:
        st.session_state['managers_df'] = read_table("managers")
    if 'customers_df' not in st.session_state:
        st.session_state['customers_df'] = read_table("customers")
    if 'products_df' not in st.session_state:
        st.session_state['products_df'] = read_table("ProductsOnWebsite")
    if 'orders_df' not in st.session_state:
        st.session_state['orders_df'] = read_table("Orders")
    model, demand_model, le_product = load_model("xgb_model.json", "best_random_forest_model_with_lags.pkl", "label_encoder.pkl")
    
    if st.session_state['logged_in']:
        if st.session_state['user_type'] == "Manager":
            manager_welcome_page(st.session_state['products_df'])
        elif st.session_state['user_type'] == "Customer":
            if st.session_state['view_mode'] == 'checkout':
                checkout()
            else:
                welcome_page()
    elif st.session_state['page'] == "Registration":
        registration_page(st.session_state['customers_df'])
    else:
        login_page()
