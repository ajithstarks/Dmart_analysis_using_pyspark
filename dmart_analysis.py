# %%
import os
import findspark
from dotenv import load_dotenv


# Set SPARK_HOME if not already set
os.environ['SPARK_HOME'] = '/home/ajith/spark'  
findspark.init()

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, sum, when


# %%
# Loading environment variables
load_dotenv()
data_path = os.getenv('DOWNLOAD_PATH')

# %%
def create_spark_session():
    """Create a Spark session."""
    spark = SparkSession.builder \
        .appName("Dmart Analysis") \
        .getOrCreate()
    return spark

# %%
def load_data(spark, data_path):
    """Load data into PySpark DataFrames."""
    products_df = spark.read.csv(f"{data_path}/Product.csv", header=True, inferSchema=True)
    sales_df = spark.read.csv(f"{data_path}/Sales.csv", header=True, inferSchema=True)
    customers_df = spark.read.csv(f"{data_path}/Customer.csv", header=True, inferSchema=True)
    return products_df, sales_df, customers_df

# %%
def clean_data(products_df, sales_df, customers_df):
    """Perform data cleaning and transformation."""
    # Rename columns for consistency if needed
    products_df = products_df.withColumnRenamed("Product ID", "product_id")\
        .withColumnRenamed("Sub-Category", "sub_category")\
        .withColumnRenamed("Product Name","product_name")
    sales_df = sales_df.withColumnRenamed("Order Line", "order_line")\
        .withColumnRenamed("Order ID","order_id")\
        .withColumnRenamed("Order Date","order_date")\
        .withColumnRenamed("Ship Date","ship_date")\
        .withColumnRenamed("Ship Mode","ship_mode")\
        .withColumnRenamed("Customer ID","customer_id")\
        .withColumnRenamed("Product ID","product_id")
    customers_df = customers_df.withColumnRenamed("Customer ID", "customer_id")\
        .withColumnRenamed("Customer Name","customer_name")\
        .withColumnRenamed("Postal Code","postal_code")

    # Handle missing values
    products_df = products_df.na.fill("Unknown")
    sales_df = sales_df.na.fill(0)
    customers_df = customers_df.na.fill({"Age": 0})
    
    return products_df, sales_df, customers_df

# %%
def join_data(products_df, sales_df, customers_df):
    """Join the DataFrames on relevant keys."""
    sales_with_product = sales_df.join(products_df, "product_id", "inner")
    full_data = sales_with_product.join(customers_df, "customer_id", "inner")
    return full_data

# %%
def perform_analysis(full_data):
    """Perform data analysis and querying."""
    # 1. Total sales for each product category
    total_sales_category = full_data.groupBy("sub_category").agg(sum("Sales").alias("total_sales"))
    print("1. Total sales for each product category")
    total_sales_category.show()

    # 2. Customer with the highest number of purchases
    top_customer = full_data.groupBy("customer_id").agg(count("order_id").alias("purchase_count")) \
        .orderBy(col("purchase_count").desc()).first()
    print("2. Customer with the highest number of purchases")
    print(f"Customer with highest purchases: {top_customer}")

    # 3. Average discount given on sales across all products
    avg_discount = full_data.agg(avg("Discount").alias("average_discount"))
    print("3. Average discount given on sales across all products")
    avg_discount.show()

    # 4. Unique products sold in each region
    unique_products_region = full_data.groupBy("Region").agg(count("product_id").alias("unique_products"))
    print("4. Unique products sold in each region")
    unique_products_region.show()

    # 5. Total profit generated in each state
    total_profit_state = full_data.groupBy("State").agg(sum("Profit").alias("total_profit"))
    print("5. Total profit generated in each state")
    total_profit_state.show()

    # 6. Product sub-category with the highest sales
    highest_sales_subcategory = full_data.groupBy("sub_category").agg(sum("Sales").alias("total_sales")) \
        .orderBy(col("total_sales").desc()).first()
    print("6. Product sub-category with the highest sales")
    print(f"Sub-category with highest sales: {highest_sales_subcategory}")

    # 7. Average age of customers in each segment
    avg_age_segment = full_data.groupBy("Segment").agg(avg("Age").alias("average_age"))
    print("7. Average age of customers in each segment")
    avg_age_segment.show()

    # 8. Orders shipped in each shipping mode
    shipped_orders = full_data.groupBy("ship_mode").agg(count("order_id").alias("total_orders"))
    print("8. Orders shipped in each shipping mode")
    shipped_orders.show()

    # 9. Total quantity of products sold in each city
    total_quantity_city = full_data.groupBy("City").agg(sum("Quantity").alias("total_quantity"))
    print("9. Total quantity of products sold in each city")
    total_quantity_city.show()

    # 10. Customer segment with the highest profit margin
    highest_profit_margin_segment = full_data.groupBy("Segment").agg(sum("Profit").alias("total_profit")) \
        .orderBy(col("total_profit").desc()).first()
    print("10. Customer segment with the highest profit margin")
    print(f"Customer segment with highest profit margin: {highest_profit_margin_segment}")

# %%
if __name__ == '__main__':
    # Define the path to your dataset
    data_path = "data"

    spark = create_spark_session()
    products_df, sales_df, customers_df = load_data(spark, data_path)
    products_df, sales_df, customers_df = clean_data(products_df, sales_df, customers_df)
    full_data = join_data(products_df, sales_df, customers_df)
    perform_analysis(full_data)

    spark.stop()

# %%



