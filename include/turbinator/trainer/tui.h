#ifndef TURBINATOR_TRAINER_TUI_H
#define TURBINATOR_TRAINER_TUI_H

#ifdef __cplusplus
extern "C" {
#endif

/* State flags exposed through the public C ABI. */
enum {
    TB_TUI_PAUSED = 1,
    TB_TUI_STOP = 2,
    TB_TUI_CHECKPOINT = 4,
    TB_TUI_EVALUATE = 8,
    TB_TUI_RECORD = 16,
    TB_TUI_BWB = 32,
    TB_TUI_CFD = 64,
    TB_TUI_INDEFINITE = 128
};

#ifdef __cplusplus
}
#endif
#endif
