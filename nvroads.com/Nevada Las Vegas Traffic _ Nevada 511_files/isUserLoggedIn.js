//Checks if the response from an ajax call includes the login page, if it does reload the page
//This is kind of why we do this : http://haacked.com/archive/2011/10/04/prevent-forms-authentication-login-page-redirect-when-you-donrsquot-want.aspx/
window.isUserLoggedIn = function (responseText) {
    if (responseText) {
        if (responseText.indexOf('form id="loginFormpage"') >= 0) {
            location.reload();
        }
    }
};