<script lang="ts">
	import { marked } from 'marked';

	export let content = '';
</script>

<div class="message border rounded-md py-1 px-2.5 my-0.25 break-words self-end bg-slate-200">
	{@html marked(content, { breaks: true, gfm: true })}
</div>

<style>
	.message {
		max-width: 80%;
	}
</style>
