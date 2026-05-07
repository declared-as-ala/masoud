import axios from "axios";

const baseURL = import.meta.env.VITE_RPI_API_URL || "http://127.0.0.1:8000";

export const api = axios.create({
  baseURL,
  timeout: 6000,
});

export const getVideoFeedUrl = () => `${baseURL}/video-feed`;
