# FROM node:20-alpine

# WORKDIR /app
# COPY package.json pnpm-lock.yaml ./
# RUN corepack enable && pnpm install

# COPY . .
# CMD ["pnpm", "dev"]
# ui/Dockerfile
# ui/Dockerfile
FROM node:20-alpine
WORKDIR /app

# 1) deps layer (MUST copy lockfile!)
COPY package.json pnpm-lock.yaml ./
RUN corepack enable && pnpm install

# 2) app code
COPY . .

# 3) run dev server on all interfaces
CMD ["pnpm", "dev", "-H", "0.0.0.0", "-p", "3000"]