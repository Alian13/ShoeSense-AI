import { createRouter, createWebHistory } from "vue-router";
import Upload from "../views/upload.vue";

const routes = [
  {
    path: "/",
    name: "Upload",
    component: Upload,
  },
];

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
});

export default router;
