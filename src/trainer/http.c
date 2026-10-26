#include "turbinator/trainer/http.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET tb_socket;
#define TB_SOCKET_INVALID INVALID_SOCKET
#define tb_close_socket closesocket
#else
#include <arpa/inet.h>
#include <dirent.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int tb_socket;
#define TB_SOCKET_INVALID (-1)
#define tb_close_socket close
#endif

static void tb_response(tb_socket fd, int code, const char *body) {
    char header[512];
    size_t n = strlen(body);
    int h = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\nContent-Type: application/json; charset=utf-8\r\n"
        "Content-Length: %zu\r\nAccess-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\nConnection: close\r\n\r\n",
        code, code == 200 ? "OK" : "Error", n);
    if (h <= 0) return;
    send(fd, header, h, 0);
    for (size_t off = 0; off < n;) {
        int sent = send(fd, body + off, (int)(n - off), 0);
        if (sent <= 0) break;
        off += (size_t)sent;
    }
}

static int tb_safe_name(const char *name) {
    if (!name || !*name || strlen(name) > 200) return 0;
    for (const char *p = name; *p; ++p) {
        if (!( (*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
               (*p >= '0' && *p <= '9') || *p == '_' || *p == '-' || *p == '.')) return 0;
    }
    return strstr(name, "..") == NULL;
}

static char *tb_read_file(const char *path, size_t limit) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0 || (size_t)n > limit) { fclose(f); return NULL; }
    rewind(f);
    char *out = (char *)malloc((size_t)n + 1);
    if (!out) { fclose(f); return NULL; }
    size_t got = fread(out, 1, (size_t)n, f);
    fclose(f);
    out[got] = 0;
    return out;
}

static void tb_write_file(const char *path, const char *body, size_t size) {
    char tmp[1200];
    if (snprintf(tmp, sizeof(tmp), "%s.tmp", path) >= (int)sizeof(tmp)) return;
    FILE *f = fopen(tmp, "wb");
    if (!f) return;
    fwrite(body, 1, size, f);
    fclose(f);
#ifdef _WIN32
    remove(path);
#endif
    rename(tmp, path);
}

static int tb_json_field(const char *body, const char *field, char *out, size_t cap) {
    char key[80];
    snprintf(key, sizeof(key), "\"%s\"", field);
    const char *p = strstr(body, key);
    if (!p) return 0;
    p = strchr(p + strlen(key), ':');
    if (!p || !(p = strchr(p, '"'))) return 0;
    ++p;
    size_t n = 0;
    while (p[n] && p[n] != '"' && n + 1 < cap) { out[n] = p[n]; ++n; }
    out[n] = 0;
    return n > 0;
}

static void tb_handle_client(tb_http_server *s, tb_socket fd) {
    char request[32768];
    int n = recv(fd, request, (int)sizeof(request) - 1, 0);
    if (n <= 0) return;
    request[n] = 0;
    char method[12] = {0}, route[256] = {0};
    if (sscanf(request, "%11s %255s", method, route) != 2) { tb_response(fd, 400, "{\"error\":\"bad_request\"}"); return; }
    char *query = strchr(route, '?');
    if (query) *query = 0;
    char path[1400];
    if (strcmp(method, "OPTIONS") == 0) { tb_response(fd, 200, "{}"); return; }
    if (strcmp(method, "GET") == 0) {
        if (strcmp(route, "/api/v1/status") == 0 || strcmp(route, "/api/status") == 0 ||
            strcmp(route, "/api/v1/mesh_sync") == 0) {
            snprintf(path, sizeof(path), "%s/live_status.json", s->season_dir);
            char *body = tb_read_file(path, 1024 * 1024);
            if (!body) { snprintf(path, sizeof(path), "%s/ipc_state.json", s->season_dir); body = tb_read_file(path, 1024 * 1024); }
            tb_response(fd, 200, body ? body : "{\"state\":\"IDLE\"}");
            free(body);
            return;
        }
        if (strcmp(route, "/api/v1/replays") == 0 || strcmp(route, "/api/replays") == 0) {
            /* The native recorder writes JSONL trajectories; list entries when present. */
            char body[8192]; size_t used = 0;
            used += (size_t)snprintf(body, sizeof(body), "{\"replays\":[");
            int first = 1;
#ifdef _WIN32
            WIN32_FIND_DATAA data;
            snprintf(path, sizeof(path), "%s/replays/*.jsonl", s->season_dir);
            HANDLE find = FindFirstFileA(path, &data);
            if (find != INVALID_HANDLE_VALUE) {
                do {
                    if (used + strlen(data.cFileName) + 8 >= sizeof(body)) break;
                    used += (size_t)snprintf(body + used, sizeof(body) - used, "%s\"%s\"", first ? "" : ",", data.cFileName);
                    first = 0;
                } while (FindNextFileA(find, &data));
                FindClose(find);
            }
#else
            snprintf(path, sizeof(path), "%s/replays", s->season_dir);
            DIR *dir = opendir(path);
            if (dir) {
                struct dirent *entry;
                while ((entry = readdir(dir))) {
                    if (!strstr(entry->d_name, ".jsonl") || used + strlen(entry->d_name) + 8 >= sizeof(body)) continue;
                    used += (size_t)snprintf(body + used, sizeof(body) - used, "%s\"%s\"", first ? "" : ",", entry->d_name);
                    first = 0;
                }
                closedir(dir);
            }
#endif
            snprintf(body + used, sizeof(body) - used, "]}");
            tb_response(fd, 200, body);
            return;
        }
        if (strncmp(route, "/api/v1/replays/download/", 25) == 0) {
            const char *name = route + 25;
            if (!tb_safe_name(name)) { tb_response(fd, 400, "{\"error\":\"invalid_name\"}"); return; }
            snprintf(path, sizeof(path), "%s/replays/%s", s->season_dir, name);
            char *body = tb_read_file(path, 4 * 1024 * 1024);
            tb_response(fd, body ? 200 : 404, body ? body : "{\"error\":\"not_found\"}");
            free(body);
            return;
        }
        tb_response(fd, 404, "{\"error\":\"not_found\"}");
        return;
    }
    if (strcmp(method, "POST") == 0 && (strcmp(route, "/api/v1/control") == 0 || strcmp(route, "/api/control") == 0)) {
        char *body = strstr(request, "\r\n\r\n");
        if (!body) { tb_response(fd, 400, "{\"error\":\"missing_body\"}"); return; }
        body += 4;
        char state[80] = {0};
        char control[512];
        if (tb_json_field(body, "training_control", state, sizeof(state)))
            snprintf(control, sizeof(control), "{\"requested_state\":\"%s\"}", state);
        else if (strstr(body, "\"paused\":true"))
            snprintf(control, sizeof(control), "{\"requested_state\":\"PAUSED\"}");
        else if (strstr(body, "\"paused\":false"))
            snprintf(control, sizeof(control), "{\"requested_state\":\"TRAINING_ACTIVE\"}");
        else if (strstr(body, "\"indefinite_training\":true"))
            snprintf(control, sizeof(control), "{\"indefinite_training\":true}");
        else if (strstr(body, "\"indefinite_training\":false"))
            snprintf(control, sizeof(control), "{\"indefinite_training\":false}");
        else { tb_response(fd, 400, "{\"error\":\"unknown_control\"}"); return; }
        snprintf(path, sizeof(path), "%s/training_control.json", s->season_dir);
        tb_write_file(path, control, strlen(control));
        tb_response(fd, 200, "{\"ok\":true}");
        return;
    }
    tb_response(fd, 404, "{\"error\":\"not_found\"}");
}

int tb_http_start(tb_http_server *s, const char *season_dir, const char *host, unsigned port) {
    if (!s || !season_dir || port == 0 || port > 65535) return 0;
    memset(s, 0, sizeof(*s));
#ifdef _WIN32
    WSADATA data;
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return 0;
#endif
    tb_socket fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == TB_SOCKET_INVALID) return 0;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    if (inet_pton(AF_INET, host && *host ? host : "127.0.0.1", &addr.sin_addr) != 1) {
        tb_close_socket(fd); return 0;
    }
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0 || listen(fd, 8) != 0) {
        tb_close_socket(fd); return 0;
    }
    s->socket_handle = (intptr_t)fd;
    snprintf(s->season_dir, sizeof(s->season_dir), "%s", season_dir);
    s->port = port;
    s->running = 1;
    return 1;
}

void tb_http_poll(tb_http_server *s) {
    if (!s || !s->running) return;
    tb_socket fd = (tb_socket)s->socket_handle;
    fd_set read_set;
    struct timeval tv = {0, 0};
    FD_ZERO(&read_set); FD_SET(fd, &read_set);
    if (select((int)fd + 1, &read_set, NULL, NULL, &tv) <= 0) return;
    struct sockaddr_in client_addr;
#ifdef _WIN32
    int len = sizeof(client_addr);
#else
    socklen_t len = sizeof(client_addr);
#endif
    tb_socket client = accept(fd, (struct sockaddr *)&client_addr, &len);
    if (client == TB_SOCKET_INVALID) return;
    tb_handle_client(s, client);
    tb_close_socket(client);
}

void tb_http_stop(tb_http_server *s) {
    if (!s || !s->running) return;
    tb_close_socket((tb_socket)s->socket_handle);
    s->running = 0;
#ifdef _WIN32
    WSACleanup();
#endif
}
