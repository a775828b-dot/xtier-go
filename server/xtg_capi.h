/* The C interface the Android app (JNI) and other hosts call: one server per process. */
#ifndef XTG_CAPI_H
#define XTG_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

/* Start loading and serving. config_json: {"model_dir", "host", "port", "served", "api_key", "threads", "cpus",
   "options": {...pack options...}, "template_kwargs": {...}}. Returns 0, or -1 with a message in err. */
int xtg_start(const char* config_json, char* err, int err_len);

/* Stop the server and free the model (blocks until the current token is done). */
int xtg_stop(void);

/* The status JSON (as GET /xtg/status). Returns its length; writes at most len - 1 bytes and a terminating 0. */
int xtg_status(char* buf, int len);

/* Memory the options would take for this model directory, without loading it (JSON, as xtg-cli estimate). */
int xtg_estimate(const char* model_dir, const char* options_json, char* buf, int len);

const char* xtg_version(void);

#ifdef __cplusplus
}
#endif

#endif
