### BUILD ###

FROM node:14.15-stretch AS build
ARG IDLP_HOST=localhost
ENV IDLP_HOST=${IDLP_HOST}
RUN mkdir /code
WORKDIR /code
COPY . .
RUN npm install
RUN npm run build -- --prod --output-hashing all

### RUN ###

FROM nginx:stable-alpine
COPY ./ui.nginx.conf /etc/nginx/conf.d/ui.nginx.conf
RUN mkdir /code
WORKDIR /code

COPY --from=build /code/dist/ui /usr/share/nginx/html
