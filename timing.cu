#include <helper_timer.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#define TIMING

/*  startTimer
 *
 *  timer - id of timer
 *
 *  Creates a new timer from the parameter.
 *  Starts the timer.
 */

void startTimer(StopWatchInterface **timer) {
    sdkCreateTimer(timer);
    checkCudaErrors( cudaThreadSynchronize() );
    sdkStartTimer(timer);
}


/*  endTimer
 *
 *  timer - id of previously started timer.
 *
 *  Waits for device to finish computing
 *  and then stops the timer.  Returns the
 *  duration of the timer.
 */

float endTimer(StopWatchInterface **timer) {

    checkCudaErrors( cudaThreadSynchronize() );
    sdkStopTimer(timer);
    float device_time = sdkGetTimerValue(timer);
    sdkDeleteTimer(timer);

    return device_time;
}
