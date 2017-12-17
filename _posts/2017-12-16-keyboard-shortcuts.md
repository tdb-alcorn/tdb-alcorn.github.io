---
layout: post
title:  "Some funky keyboard shortcuts"
date:   2017-12-16 15:59:53 -0800
excerpt: In which your correspondent finds true love
---
In this notebook, you'll get some practice using keyboard shortcuts. These are key to becoming proficient at using notebooks and will greatly increase your work speed.

First up, switching between edit mode and command mode. Edit mode allows you to type into cells while command mode will use key presses to execute commands such as creating new cells and openning the command palette. When you select a cell, you can tell which mode you're currently working in by the color of the box around the cell. In edit mode, the box and thick left border are colored green. In command mode, they are colored blue. Also in edit mode, you should see a cursor in the cell itself.

By default, when you create a new cell or move to the next one, you'll be in command mode. To enter edit mode, press Enter/Return. To go back from edit mode to command mode, press Escape.

> **Exercise:** Click on this cell, then press Enter + Shift to get to the next cell. Switch between edit and command mode a few times.


```python
# mode practice
```

## Help with commands

If you ever need to look up a command, you can bring up the list of shortcuts by pressing `H` in command mode. The keyboard shortcuts are also available above in the Help menu. Go ahead and try it now.

## Creating new cells

One of the most common commands is creating new cells. You can create a cell above the current cell by pressing `A` in command mode. Pressing `B` will create a cell below the currently selected cell.

# haha!

$$
x = \frac{y}{z}
$$

> **Exercise:** Create a cell above this cell using the keyboard command.

> **Exercise:** Create a cell below this cell using the keyboard command.

# foo bar asdf

## Switching between Markdown and code

With keyboard shortcuts, it is quick and simple to switch between Markdown and code cells. To change from Markdown to a code cell, press `Y`. To switch from code to Markdown, press `M`.

> **Exercise:** Switch the cell below between Markdown and code cells.


```python
## Practice here

def fibo(n): # Recursive Fibonacci sequence!
    if n == 0:
        return 0
    elif n == 1:
        return 1
    return fibo(n-1) + fibo(n-2)

```

## Line numbers

A lot of times it is helpful to number the lines in your code for debugging purposes. You can turn on numbers by  pressing `L` (in command mode of course) on a code cell.

> **Exercise:** Turn line numbers on and off in the above code cell.

## Deleting cells

Deleting cells is done by pressing `D` twice in a row so `D`, `D`. This is to prevent accidently deletions, you have to press the button twice!

> **Exercise:** Delete the cell below.

## Saving the notebook

Notebooks are autosaved every once in a while, but you'll often want to save your work between those times. To save the book, press `S`. So easy!

## The Command Palette

You can easily access the command palette by pressing Shift + Control/Command + `P`. 

> **Note:** This won't work in Firefox and Internet Explorer unfortunately. There is already a keyboard shortcut assigned to those keys in those browsers. However, it does work in Chrome and Safari.

This will bring up the command palette where you can search for commands that aren't available through the keyboard shortcuts. For instance, there are buttons on the toolbar that move cells up and down (the up and down arrows), but there aren't corresponding keyboard shortcuts. To move a cell down, you can open up the command palette and type in "move" which will bring up the move commands.

> **Exercise:** Use the command palette to move the cell below down one position.


```python
# below this cell
```


```python
# Move this cell down
```

## Finishing up

There is plenty more you can do such as copying, cutting, and pasting cells. I suggest getting used to using the keyboard shortcuts, you’ll be much quicker at working in notebooks. When you become proficient with them, you'll rarely need to move your hands away from the keyboard, greatly speeding up your work.

Remember, if you ever need to see the shortcuts, just press `H` in command mode.

