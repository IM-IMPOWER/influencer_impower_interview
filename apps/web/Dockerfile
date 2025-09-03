# AIDEV-NOTE: 250903170011 Production Next.js Dockerfile with multi-stage build
# Optimized for production deployment on Vercel/Railway

FROM node:18-alpine AS base

# AIDEV-NOTE: Install dependencies only when needed
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

# AIDEV-NOTE: Copy package files
COPY package.json bun.lockb* package-lock.json* yarn.lock* pnpm-lock.yaml* ./
RUN \
  if [ -f yarn.lock ]; then yarn --frozen-lockfile; \
  elif [ -f package-lock.json ]; then npm ci; \
  elif [ -f pnpm-lock.yaml ]; then corepack enable pnpm && pnpm i --frozen-lockfile; \
  elif [ -f bun.lockb ]; then npm install -g bun && bun install --frozen-lockfile; \
  else echo "Lockfile not found." && exit 1; \
  fi

# AIDEV-NOTE: Build stage
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# AIDEV-NOTE: Build the application
ENV NEXT_TELEMETRY_DISABLED 1
ENV NODE_ENV production

RUN \
  if [ -f yarn.lock ]; then yarn run build; \
  elif [ -f package-lock.json ]; then npm run build; \
  elif [ -f pnpm-lock.yaml ]; then corepack enable pnpm && pnpm run build; \
  elif [ -f bun.lockb ]; then bun run build; \
  else npm run build; \
  fi

# AIDEV-NOTE: Production image
FROM base AS production
WORKDIR /app

# AIDEV-NOTE: Create non-root user
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# AIDEV-NOTE: Copy built application
COPY --from=builder /app/public ./public

# Set the correct permission for prerender cache
RUN mkdir .next
RUN chown nextjs:nodejs .next

# AIDEV-NOTE: Copy built files with correct permissions
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# AIDEV-NOTE: Switch to non-root user
USER nextjs

# AIDEV-NOTE: Expose port
EXPOSE 3000

ENV PORT 3000
ENV HOSTNAME "0.0.0.0"
ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

# AIDEV-NOTE: Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/api/health || exit 1

# AIDEV-NOTE: Start the application
CMD ["node", "server.js"]