""" PageGenerator.py handles the first portion of the project, taking in the
    directory, constructing the html, and passing it back to PNGViewer.sh to
    open.
    Created by Michael Stoneman in July of 2015.
"""

import os, sys, webbrowser

def getHeader(HTMLstring):
    """Opens the header file and reads it into the string."""
    header = open(os.path.dirname(os.path.realpath(__file__))+"/html/header.html", "r")
    HTMLstring += header.read()
    header.close()
    return HTMLstring
    
def getBody(HTMLstring, directory, png_list):
    """Generates the body of the HTML from the provided PNG files."""
        
    # First, ensure the slash
    if directory[-1] != "/":
        directory += "/"
    
    # Next, we generate all the rows but the last one...
    while len(png_list) > 4:
        HTMLstring += '<div class="row">'
        for i in range(4):
            HTMLstring += ('''<div class="col-xs-3 imgbox">
                <img class="img-responsive" src="''' + directory
                           + png_list[i] + '''" /><h5 class="center">''' + png_list[i]
                           + "</h5></div>")
        HTMLstring += "</div>"
        png_list = png_list[4:]
    
    # We obtain the last row by popping what remains.
    HTMLstring += '<div class="row">'
    while len(png_list) > 0:
        png_file = png_list.pop(0)
        HTMLstring +=('''<div class="col-xs-3 imgbox">
            <img class="img-responsive" src="''' + directory
                      + png_file + '''" /><h5 class="center">''' + png_file
                      + "</h5></div>")
    HTMLstring += "</div>"
    return HTMLstring

def getFooter(HTMLstring):
    """Opens the footer file and reads it into the string."""
    footer = open(os.path.dirname(os.path.realpath(__file__))+"/html/footer.html", "r")
    HTMLstring += footer.read()
    footer.close()
    return HTMLstring


class PNGViewer:

    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.png_list = []
        self.url = os.path.dirname(os.path.realpath(__file__))+"/manual_dwi_correction.html"
        
        for filepath in os.listdir(self.img_dir):
            if filepath[:2] != "._" and filepath[-4:] == ".png":
                self.png_list.append(filepath)

        if len(self.png_list) < 1:
            print("Directory must contain at least one .png file!")
            sys.exit(1)

        # Makes the HTML.
        self.webpage = getHeader("")
        self.webpage = getBody(self.webpage, self.img_dir, self.png_list)
        self.webpage = getFooter(self.webpage)

        # Writes the HTML to a file.
        self.HTML_file = open(self.url, "w")
        self.HTML_file.write(self.webpage)
        self.HTML_file.close()

    def runPNGViewer(self):
        # open a public URL, in this case, the webbrowser docs
        webbrowser.open("file://"+self.url, new=2, autoraise=True )
        
    def cleanupURL(self):
    	os.remove(self.url)
