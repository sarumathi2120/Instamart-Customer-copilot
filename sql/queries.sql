-- Q1: WHERE filters orders, HAVING filters customer totals
SELECT customer_id, SUM(order_amount) AS total_spend
FROM orders WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;
-- Q2: city by category revenue
SELECT c.city, p.category, ROUND(SUM(o.order_amount),2) AS revenue
FROM orders o JOIN customers c ON c.customer_id=o.customer_id
JOIN products p ON p.product_id=o.product_id
WHERE o.status='completed'
GROUP BY c.city,p.category ORDER BY c.city,p.category;
-- Q3: scalar subquery
SELECT * FROM products WHERE price_inr>(SELECT AVG(price_inr) FROM products) ORDER BY price_inr DESC;
-- Q4: customers without completed orders
SELECT c.customer_id, c.city FROM customers c LEFT JOIN orders o
ON c.customer_id=o.customer_id AND o.status='completed'
WHERE o.order_id IS NULL ORDER BY c.customer_id;
-- Q5: CTEs total spend and average
WITH spend AS (SELECT customer_id, SUM(order_amount) AS total FROM orders WHERE status='completed' GROUP BY customer_id),
avg_spend AS (SELECT AVG(total) AS mean_spend FROM spend)
SELECT s.customer_id,s.total,ROUND(a.mean_spend,2) AS avg_spend FROM spend s CROSS JOIN avg_spend a
WHERE s.total>a.mean_spend ORDER BY s.total DESC;
-- Q6: rank by city
WITH spending AS (
 SELECT c.customer_id,c.city,SUM(o.order_amount) AS total_spend
 FROM orders o JOIN customers c ON c.customer_id=o.customer_id
 WHERE o.status='completed' GROUP BY c.customer_id,c.city)
SELECT *, RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS rank_in_city,
 DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS dense_rank_in_city,
 ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC) AS row_number_in_city
FROM spending ORDER BY city,rank_in_city;
-- Q7: daily cumulative revenue
WITH daily AS (SELECT SUBSTR(order_ts,1,10) AS day, SUM(order_amount) AS revenue
 FROM orders WHERE status='completed' GROUP BY SUBSTR(order_ts,1,10))
SELECT day,ROUND(revenue,2) AS revenue,
 ROUND(SUM(revenue) OVER(ORDER BY day),2) AS cumulative_revenue FROM daily ORDER BY day;
