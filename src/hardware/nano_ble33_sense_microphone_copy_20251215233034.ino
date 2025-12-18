#include <Arduino.h>
#include <Bird-Sound_inferencing.h>
#include "bird_samples.h"

static size_t current_clip = 0;
static bool debug_nn = false;

static int wav_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    for (size_t i = 0; i < length; i++) {
        out_ptr[i] = audio_clips[current_clip][offset + i] / 32768.0f;
    }
    return 0;
}

void setup() {
    Serial.begin(115200);
    while (!Serial);

    ei_printf("\nEdge Impulse WAV batch inference\n");

    ei_printf("Expected samples: %d\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
    ei_printf("Clip length: %d\n", AUDIO_CLIP_LENGTH);
    ei_printf("Number of clips: %d\n\n", NUM_AUDIO_CLIPS);

    if (AUDIO_CLIP_LENGTH != EI_CLASSIFIER_RAW_SAMPLE_COUNT) {
        ei_printf("ERROR: AUDIO_CLIP_LENGTH mismatch\n");
        while (1);
    }
}

void loop() {
    ei_printf("Testing clip %d / %d : %s\n",
              current_clip + 1,
              NUM_AUDIO_CLIPS,
              audio_clip_names[current_clip]);

    signal_t signal;
    signal.total_length = AUDIO_CLIP_LENGTH;
    signal.get_data = &wav_signal_get_data;

    ei_impulse_result_t result = { 0 };

    EI_IMPULSE_ERROR rc = run_classifier(&signal, &result, debug_nn);
    if (rc != EI_IMPULSE_OK) {
        ei_printf("Classifier error (%d)\n", rc);
        return;
    }

    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        ei_printf("  %s: %.5f\n",
                  result.classification[i].label,
                  result.classification[i].value);
    }

#if EI_CLASSIFIER_HAS_ANOMALY
    ei_printf("  anomaly: %.3f\n", result.anomaly);
#endif

    ei_printf("\n");

    current_clip = (current_clip + 1) % NUM_AUDIO_CLIPS;
    delay(3000);
}
