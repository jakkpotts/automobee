/* 
    This is an api for showing a blocker and loading spinner. each call to showSpinner and hideSpinner 
    takes a process ID so that one process cannot hide the spinner of another.
*/
window.loadBlockerApi = window.loadBlockerApi || new function () {
    var currentIds = [];

    //the spinner and blocker.
    var spinner = null;

    //show a spinner for a specific id.
    this.showSpinner = function (id) {
        //fix nulls.
        id = id || 'noId';

        //id is not present in the array.
        if ($.inArray(id, currentIds) == -1) {
            currentIds.push(id);
        }

        //spinner is not currently visible.
        if (!spinner) {

            //you are not supposed to reuse bootbox modals, animate parameter causes right click on ERS map to behave like browser right click on  Google map v3.32 and 3.33
            spinner = bootbox.dialog({
                message: '<i class=\'far fa-spinner fa-spin fa-3x fa-fw\'></i><span class=\'sr-only\'>' + resources.Loading + '</span>',
                closeButton: false,
                //animate: false,
                className: 'loadingModal',
                show: true
            });
        }
    };

    //hide the spinner for a specific id.
    this.hideSpinner = function (id) {
        //fix nulls.
        id = id || 'noId';

        //remove that id from the array.
        currentIds = jQuery.grep(currentIds, function (value) {
            return value != id;
        });

        //no more items in currentIds means blocker no longer required.
        if (currentIds.length == 0 && spinner) {
            spinner.modal('hide');
            spinner = null;
        }
    };

    //hide the spinner for a specific selector.
    //this removes the spinner  without affecting the scrollbar of the body, 
    //which we want to continue to be hidden if there is another modal opened
    this.manualRemove = function (selector) {

        let modal = document.querySelector(selector);
        modal.style.visibility = 'hidden'; //hides the spinner right away, but it's still in the DOM
        modal.nextElementSibling.remove();
        modal.remove();
        spinner = null;
    }
};