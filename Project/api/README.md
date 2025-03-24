# 🚀 FastAPI Dockerized Application

This project is a FastAPI-based application for predicting body fat percentage. It includes a web form for manual input and a REST API for external integration.

## 📦 Getting Started

### 1️⃣ Run Docker (Make Sure You Are in /Project Directory)

Before running, ensure you have Docker and Docker Compose installed.

### 💻 Start the application

Run the following command for the first time (to build the images):

```sh
docker-compose up --build
```

For subsequent runs, use:

```sh
docker-compose up
```

This will start the FastAPI service inside a Docker container.

### 🔄 Rebuild API When Pulling Changes

If you pull updates from GitHub that affect the API service, rebuild the Docker containers:

```sh
docker-compose up --build
```

This ensures your application is updated with the latest changes.

## 🌐 Using the Application

### 1️⃣ Open the App

Once the application is running, go to:

🔗 **[http://localhost:8000](http://localhost:8000)**

Use the web interface to input body measurements and get predictions.

### 2️⃣ Access API Documentation

FastAPI provides an interactive Swagger UI for API testing:

🔗 **[http://localhost:8000/docs](http://localhost:8000/docs)**

This allows you to test API requests directly from the browser.

## 📡 Using API from Outside

You can use `curl` or any HTTP client (like Postman) to send requests to the API.

### 🔹 Example Request

```sh
curl -X POST "http://localhost:8000/predict/"      -H "accept: application/json"      -H "Content-Type: application/x-www-form-urlencoded"      -d "abdomen=23&hip=23&weight=23&thigh=23&knee=23&biceps=3&neck=23"
```

### 🔹 Example Response

```json
{
  "prediction": -23.959071614389472
}
```

You can modify the parameters (`abdomen`, `hip`, `weight`, etc.) to get different predictions.

## ⚙ Advanced Usage

### 📌 Run Docker in Detached Mode

To run the containers in the background, use:

```sh
docker-compose up -d
```

To stop running containers:

```sh
docker-compose down
```

### 📌 Check Running Containers

To list all running containers:

```sh
docker ps
```

### 📌 View API Logs

To debug issues or see FastAPI logs:

```sh
docker-compose logs -f
```

### 📌 Restart the API Container

```sh
docker-compose restart
```

## 🚀 Deployment (Expose API Publicly)

By default, the API is only available locally. To expose it to the outside world, make sure your server’s firewall allows traffic on port 8000.

### Run FastAPI on a Public Host

Modify the `docker-compose.yml` file to ensure the service listens on all interfaces:

```yaml
ports:
  - "8000:8000"
```

Then restart the container:

```sh
docker-compose down
docker-compose up -d
```

Now, the API should be accessible from other devices via:

🔗 **http://YOUR_SERVER_IP:8000**

## 📄 API Endpoints

| Method | Endpoint     | Description                                  |
|--------|-------------|----------------------------------------------|
| GET    | `/`         | Returns the web form                        |
| POST   | `/predict/` | Sends body measurements and returns a body fat prediction |

## 🛠 Troubleshooting

### 🔹 Docker Image Not Updating

Try:

```sh
docker-compose down
docker system prune -a
docker-compose up --build
```

### 🔹 API Not Working?

Check container logs:

```sh
docker-compose logs -f
```

Ensure port `8000` is not blocked by a firewall.

## 🎯 Next Steps

✅ Add frontend styling for a better UI.

✅ Deploy the API to a cloud server.

✅ Integrate with a database for storing past predictions.

## 💡 Contributing

Feel free to fork this project and submit a pull request. If you find a bug or have a feature request, open an issue.

🚀 **Happy Coding!** 🚀