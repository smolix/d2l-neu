-- Injected cell outputs (code blocks without a language) become the framed
-- Output verbatim box of d2l-beamer.tex; code cells keep pandoc's highlighting.
function CodeBlock(el)
  if #el.classes == 0 then
    return pandoc.RawBlock('latex',
      '\\begin{Output}\n' .. el.text .. '\n\\end{Output}')
  end
end
