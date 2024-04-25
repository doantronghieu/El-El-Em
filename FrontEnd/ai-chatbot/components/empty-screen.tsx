import { UseChatHelpers } from 'ai/react'

import { Button } from '@/components/ui/button'
import { ExternalLink } from '@/components/external-link'
import { IconArrowRight } from '@/components/ui/icons'

const exampleMessages = [
  {
    heading: 'Explain technical concepts',
    message: `What is a "serverless function"?`
  },
  {
    heading: 'Summarize an article',
    message: 'Summarize the following article for a 2nd grader: \n'
  },
  {
    heading: 'Draft an email',
    message: `Draft an email to my boss about the following: \n`
  }
]

export function EmptyScreen() {
  return (
    <div className="max-w-2xl px-4 mx-auto">
      <div className="flex flex-col gap-2 p-8 border rounded-lg bg-background">
        <h1 className="text-lg font-semibold">
          Welcome to My AI Chatbot!
        </h1>
       
      </div>
    </div>
  )
}
