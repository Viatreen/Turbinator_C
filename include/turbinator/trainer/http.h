#ifndef TURBINATOR_TRAINER_HTTP_H
#define TURBINATOR_TRAINER_HTTP_H
#include <stdint.h>

typedef struct tb_http_server {
    intptr_t socket_handle;
    char season_dir[1024];
    unsigned port;
    int running;
} tb_http_server;

int tb_http_start(tb_http_server *server, const char *season_dir, const char *host, unsigned port);
void tb_http_poll(tb_http_server *server);
void tb_http_stop(tb_http_server *server);
#endif
